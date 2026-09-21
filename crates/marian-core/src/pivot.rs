use crate::{BackendError, BackendInfo, TranslationBackend, TranslationInput, TranslationOutput};

/// Two models owned by the same scheduler thread. Each source row remains an
/// independent sequence across both stages; no textual numbering is needed.
pub struct PivotBackend<A, B> {
    first: A,
    second: B,
    source: String,
    pivot: String,
    target: String,
}

impl<A: TranslationBackend, B: TranslationBackend> PivotBackend<A, B> {
    pub fn new(first: A, second: B, source: &str, pivot: &str, target: &str) -> Self {
        Self {
            first,
            second,
            source: source.into(),
            pivot: pivot.into(),
            target: target.into(),
        }
    }
}

impl<A: TranslationBackend, B: TranslationBackend> TranslationBackend for PivotBackend<A, B> {
    fn info(&self) -> BackendInfo {
        let mut info = self.second.info();
        info.model = format!(
            "{} -> {} (via {})",
            self.first.info().model,
            info.model,
            self.pivot
        );
        info.supports_batching &= self.first.info().supports_batching;
        info
    }

    fn is_ready(&self) -> bool {
        self.first.is_ready() && self.second.is_ready()
    }

    fn translate_batch(
        &mut self,
        inputs: &[TranslationInput],
    ) -> Result<Vec<TranslationOutput>, BackendError> {
        self.translate_batch_with_repetitions(inputs, &vec![1; inputs.len()])
    }

    fn translate_batch_with_repetitions(
        &mut self,
        inputs: &[TranslationInput],
        repetitions: &[usize],
    ) -> Result<Vec<TranslationOutput>, BackendError> {
        let Some(head) = inputs.first() else {
            return Ok(vec![]);
        };
        if inputs.len() != repetitions.len()
            || inputs
                .iter()
                .any(|i| i.source_lang != head.source_lang || i.target_lang != head.target_lang)
        {
            return Err(BackendError::InvalidInput(
                "pivot batch must have one language direction and matching repetition counts"
                    .into(),
            ));
        }
        if head.source_lang == self.source && head.target_lang == self.pivot {
            return self
                .first
                .translate_batch_with_repetitions(inputs, repetitions);
        }
        if head.source_lang == self.pivot && head.target_lang == self.target {
            return self
                .second
                .translate_batch_with_repetitions(inputs, repetitions);
        }
        if head.source_lang != self.source || head.target_lang != self.target {
            return Err(BackendError::UnsupportedDirection(format!(
                "{} -> {}; available: {} -> {}, {} -> {}, {} -> {}",
                head.source_lang,
                head.target_lang,
                self.source,
                self.pivot,
                self.pivot,
                self.target,
                self.source,
                self.target
            )));
        }
        let first_inputs: Vec<_> = inputs
            .iter()
            .map(|i| {
                let mut next = i.clone();
                next.target_lang = self.pivot.clone();
                // A short target budget must not truncate the intermediate English.
                next.max_output_tokens = i.max_output_tokens.max(512);
                next
            })
            .collect();
        let intermediate = self
            .first
            .translate_batch_with_repetitions(&first_inputs, repetitions)?;
        if intermediate.len() != inputs.len()
            || intermediate.iter().any(|x| x.text.trim().is_empty())
        {
            return Err(BackendError::Inference(
                "pivot stage returned missing translations".into(),
            ));
        }
        let second_inputs: Vec<_> = intermediate
            .iter()
            .zip(inputs)
            .map(|(output, input)| {
                let mut next = input.clone();
                next.text = output.text.clone();
                next.source_lang = self.pivot.clone();
                next
            })
            .collect();
        let mut translated = self
            .second
            .translate_batch_with_repetitions(&second_inputs, repetitions)?;
        if translated.len() != inputs.len() || translated.iter().any(|x| x.text.trim().is_empty()) {
            return Err(BackendError::Inference(
                "target stage returned missing translations".into(),
            ));
        }
        for (output, first) in translated.iter_mut().zip(intermediate) {
            output.input_tokens = first.input_tokens;
            output.score = None; // a second-stage score is not a JA -> ZH score
        }
        Ok(translated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EchoBackend;

    struct Stage {
        from: &'static str,
        to: &'static str,
        missing: bool,
    }
    impl TranslationBackend for Stage {
        fn info(&self) -> BackendInfo {
            EchoBackend.info()
        }
        fn translate_batch(
            &mut self,
            inputs: &[TranslationInput],
        ) -> Result<Vec<TranslationOutput>, BackendError> {
            assert!(
                inputs
                    .iter()
                    .all(|i| i.source_lang == self.from && i.target_lang == self.to)
            );
            if self.missing {
                return Ok(vec![]);
            }
            Ok(inputs
                .iter()
                .map(|i| TranslationOutput {
                    text: format!("{}[{}]", self.to, i.text),
                    score: Some(1.0),
                    input_tokens: 3,
                    output_tokens: 4,
                })
                .collect())
        }
    }
    fn backend(missing: bool) -> PivotBackend<Stage, Stage> {
        PivotBackend::new(
            Stage {
                from: "ja",
                to: "en",
                missing,
            },
            Stage {
                from: "en",
                to: "zh",
                missing: false,
            },
            "ja",
            "en",
            "zh",
        )
    }
    #[test]
    fn pivot_preserves_rows_and_direct_routes() {
        let mut b = backend(false);
        let results = b
            .translate_batch(&[
                TranslationInput::new("一", "ja", "zh"),
                TranslationInput::new("二", "ja", "zh"),
            ])
            .unwrap();
        assert_eq!(
            results.iter().map(|o| o.text.as_str()).collect::<Vec<_>>(),
            ["zh[en[一]]", "zh[en[二]]"]
        );
        assert!(results.iter().all(|o| o.score.is_none()));
        assert_eq!(
            b.translate_batch(&[TranslationInput::new("hello", "en", "zh")])
                .unwrap()[0]
                .text,
            "zh[hello]"
        );
        assert_eq!(
            b.translate_batch(&[TranslationInput::new("一", "ja", "en")])
                .unwrap()[0]
                .text,
            "en[一]"
        );
    }
    #[test]
    fn missing_rows_and_unsupported_directions_fail_closed() {
        assert!(
            backend(true)
                .translate_batch(&[TranslationInput::new("一", "ja", "zh")])
                .is_err()
        );
        assert!(
            backend(false)
                .translate_batch(&[TranslationInput::new("一", "zh", "ja")])
                .is_err()
        );
    }
}
