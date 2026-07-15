use crate::metal_runtime::Buffer;

use super::ops::AttentionView;
use super::{EncodedBatch, MAXIMUM_POSITION, MetalEngine, checked_mul, to_u32};

struct CandidateBatch {
    width: usize,
    counts: Buffer,
    id_buffer: Buffer,
}

pub(crate) struct BatchOutput {
    pub(crate) tokens: Vec<i32>,
    pub(crate) offsets: Vec<u32>,
}

impl MetalEngine {
    pub(super) fn decode(
        &self,
        tokens: &[i32],
        offsets: &[u32],
        max_output_tokens: &[usize],
        encoded: &EncodedBatch,
    ) -> Result<BatchOutput, String> {
        let batch = offsets.len() - 1;
        let candidates = self.prepare_candidates(tokens, offsets)?;
        let states = (0..self.model.decoder.len())
            .map(|_| self.upload_buffer(&vec![0.0_f32; checked_mul(batch, self.dim)?]))
            .collect::<Result<Vec<_>, String>>()?;
        let previous_buffer = self.upload_buffer(&vec![-1_i32; batch])?;
        let mut generated = vec![Vec::<i32>::new(); batch];
        let limits = encoded
            .lengths
            .iter()
            .zip(max_output_tokens)
            .map(|(&length, &requested)| {
                requested
                    .min(length.saturating_mul(self.max_length_factor))
                    .min(MAXIMUM_POSITION)
            })
            .collect::<Vec<_>>();
        let generation_limit = limits.iter().copied().max().unwrap_or(0);
        let gpu_limits = limits
            .iter()
            .map(|&limit| to_u32(limit, "output token limit"))
            .collect::<Result<Vec<_>, _>>()?;
        let limit_buffer = self.upload_buffer(&gpu_limits)?;
        let finished_buffer = self.upload_buffer(&vec![0_u32; batch])?;
        let mut step = 0;
        let mut active_rows = batch;
        let mut completion_in_previous_submission = false;

        while step < generation_limit {
            let steps_in_submission = self.tuning.decode.submission_steps(
                active_rows,
                generation_limit - step,
                completion_in_previous_submission,
            );
            let _scratch = self.begin_scratch()?;
            let history_buffer = self.workspace.transient_upload(
                &self.runtime,
                &vec![-1_i32; checked_mul(steps_in_submission, batch)?],
            )?;
            let commands = self.commands(&format!(
                "decode[{step}..{}) active={active_rows}",
                step + steps_in_submission
            ))?;
            for history_step in 0..steps_in_submission {
                let absolute_step = step + history_step;
                let mut decoder =
                    self.decoder_input(&commands, &previous_buffer, batch, absolute_step)?;

                for (index, layer) in self.model.decoder.iter().enumerate() {
                    let projection = self.matmul(
                        &commands,
                        &decoder,
                        &layer.ssru.projection,
                        None,
                        batch,
                        checked_mul(self.dim, 2)?,
                        self.dim,
                    )?;
                    decoder = self.ssru_norm(
                        &commands,
                        &projection,
                        &layer.ssru.bf,
                        &states[index],
                        &decoder,
                        &layer.ssru.norm_scale,
                        &layer.ssru.norm_bias,
                        batch,
                    )?;

                    let query = self.matmul(
                        &commands,
                        &decoder,
                        &layer.context.query,
                        Some(&layer.context.query_bias),
                        batch,
                        self.dim,
                        self.dim,
                    )?;
                    let attended = self.attend(
                        &commands,
                        AttentionView::contiguous(&query, self.dim),
                        AttentionView {
                            buffer: &encoded.cross[index].key_value,
                            stride: self.dim * 2,
                            offset: 0,
                        },
                        AttentionView {
                            buffer: &encoded.cross[index].key_value,
                            stride: self.dim * 2,
                            offset: self.dim,
                        },
                        &encoded.length_buffer,
                        batch,
                        1,
                        encoded.source_length,
                    )?;
                    decoder = self.linear_residual_norm(
                        &commands,
                        &attended,
                        &layer.context.output.weight,
                        &layer.context.output.bias,
                        &decoder,
                        &layer.context.output.norm_scale,
                        &layer.context.output.norm_bias,
                        batch,
                        self.dim,
                    )?;
                    decoder = self.feed_forward(&commands, &decoder, &layer.ffn, batch)?;
                }

                self.select_and_advance(
                    &commands,
                    &decoder,
                    candidates.width,
                    &candidates.counts,
                    &candidates.id_buffer,
                    &previous_buffer,
                    &limit_buffer,
                    &finished_buffer,
                    &history_buffer,
                    batch,
                    absolute_step,
                    history_step,
                )?;
            }
            self.finish_commands(commands)?;
            let history = history_buffer.read::<i32>(checked_mul(steps_in_submission, batch)?)?;
            for history_step in 0..steps_in_submission {
                for row in 0..batch {
                    let token = history[history_step * batch + row];
                    if token >= 0 {
                        generated[row].push(token);
                    }
                }
            }
            step += steps_in_submission;
            let next_active_rows = finished_buffer
                .read::<u32>(batch)?
                .into_iter()
                .filter(|&value| value == 0)
                .count();
            completion_in_previous_submission = next_active_rows < active_rows;
            active_rows = next_active_rows;
            if active_rows == 0 {
                break;
            }
        }

        let mut output = BatchOutput {
            tokens: Vec::new(),
            offsets: Vec::with_capacity(batch + 1),
        };
        output.offsets.push(0);
        for sentence in generated {
            output.tokens.extend(sentence);
            output.offsets.push(
                u32::try_from(output.tokens.len())
                    .map_err(|_| "generated token count exceeds u32".to_string())?,
            );
        }
        Ok(output)
    }

    fn prepare_candidates(
        &self,
        tokens: &[i32],
        offsets: &[u32],
    ) -> Result<CandidateBatch, String> {
        let rows = offsets
            .windows(2)
            .map(|pair| {
                self.shortlist
                    .candidates(&tokens[pair[0] as usize..pair[1] as usize])
            })
            .collect::<Result<Vec<_>, _>>()?;
        let width = rows.iter().map(Vec::len).max().unwrap_or(0);
        if width == 0 {
            return Err("lexical shortlist produced an empty batch".into());
        }
        let counts = rows
            .iter()
            .map(|row| to_u32(row.len(), "candidate count"))
            .collect::<Result<Vec<_>, _>>()?;
        let mut ids = vec![0_u32; checked_mul(rows.len(), width)?];
        for (row_index, row) in rows.iter().enumerate() {
            ids[row_index * width..row_index * width + row.len()].copy_from_slice(row);
        }
        Ok(CandidateBatch {
            id_buffer: self.upload_buffer(&ids)?,
            counts: self.upload_buffer(&counts)?,
            width,
        })
    }
}
