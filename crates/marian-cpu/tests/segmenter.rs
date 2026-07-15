#[path = "../src/segmenter.rs"]
mod segmenter;

use segmenter::{SegmentError, TextSegment, segment_text};

fn word_pieces(text: &str) -> Result<usize, String> {
    Ok(text.split_whitespace().count())
}

fn scalar_pieces(text: &str) -> Result<usize, String> {
    Ok(text
        .chars()
        .filter(|character| !character.is_whitespace())
        .count())
}

fn reconstructed(text: &str, segments: &[TextSegment]) -> String {
    segments
        .iter()
        .flat_map(|segment| [segment.content(text), segment.separator(text)])
        .collect()
}

#[test]
fn short_text_stays_in_one_lossless_segment() {
    let text = "Hello world.  \r\n";
    let segments = segment_text(text, 255, word_pieces).unwrap();

    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].content(text), "Hello world.");
    assert_eq!(segments[0].separator(text), "  \r\n");
    assert_eq!(segments[0].full_range(), 0..text.len());
    assert_eq!(reconstructed(text, &segments), text);
}

#[test]
fn abbreviation_decimal_version_domain_and_email_do_not_make_false_boundaries() {
    let prefix = "Dr. Smith used e.g. version 1.2.3 at example.com and a.b@example.org. ";
    let text = format!("{prefix}{} End.", "word ".repeat(260));
    let segments = segment_text(&text, 255, word_pieces).unwrap();

    assert_eq!(segments[0].content(&text), prefix.trim_end());
    assert!(!segments.iter().any(|segment| {
        matches!(
            segment.content(&text),
            "Dr." | "e.g." | "version 1.2." | "example." | "a.b@example."
        )
    }));
    assert_eq!(reconstructed(&text, &segments), text);
    assert!(segments.iter().all(|segment| segment.source_pieces <= 255));
}

#[test]
fn sentence_quotes_cjk_terminators_crlf_and_blank_lines_are_preserved() {
    let text = format!(
        "{} \r\n\r\nHe asked, \"Ready?!\"  中文句子。\n下一句！  Final…\u{201d}",
        "word ".repeat(256)
    );
    let segments = segment_text(&text, 255, word_pieces).unwrap();

    assert_eq!(reconstructed(&text, &segments), text);
    assert!(segments.iter().all(|segment| segment.source_pieces <= 255));
    assert!(
        segments
            .iter()
            .any(|segment| segment.separator(&text) == "  \r\n\r\n")
    );
    assert!(
        segments
            .iter()
            .any(|segment| segment.content(&text).ends_with("Ready?!\""))
    );
    assert!(
        segments
            .iter()
            .any(|segment| segment.content(&text).ends_with("中文句子。"))
    );
}

#[test]
fn short_multiline_text_keeps_each_separator_outside_the_model() {
    let text = "Line one.\nLine two?\r\nLine three!";
    let segments = segment_text(text, 255, word_pieces).unwrap();

    assert_eq!(segments.len(), 3);
    assert_eq!(segments[0].separator(text), "\n");
    assert_eq!(segments[1].separator(text), "\r\n");
    assert_eq!(segments[2].separator(text), "");
    assert_eq!(reconstructed(text, &segments), text);
}

#[test]
fn seven_hundred_tokens_split_at_preferred_boundaries() {
    let text = (0..700)
        .map(|index| format!("token{index}"))
        .collect::<Vec<_>>()
        .join(" ");
    let segments = segment_text(&text, 255, word_pieces).unwrap();

    assert_eq!(segments.len(), 3);
    assert_eq!(segments[0].source_pieces, 255);
    assert_eq!(segments[1].source_pieces, 255);
    assert_eq!(segments[2].source_pieces, 190);
    assert_eq!(reconstructed(&text, &segments), text);
}

#[test]
fn falls_back_through_punctuation_whitespace_and_utf8_scalars() {
    for delimiter in [";", ":", "/", " —", " --", ","] {
        let punctuation = format!(
            "{}{delimiter} {}{delimiter} {}",
            "a".repeat(140),
            "b".repeat(140),
            "c".repeat(40)
        );
        let punctuation_segments = segment_text(&punctuation, 150, scalar_pieces).unwrap();
        assert_eq!(
            reconstructed(&punctuation, &punctuation_segments),
            punctuation,
            "delimiter={delimiter:?}"
        );
        assert!(
            punctuation_segments[0]
                .content(&punctuation)
                .ends_with(delimiter.trim_start()),
            "delimiter={delimiter:?}"
        );
    }

    let no_spaces = format!(
        "{}{}{}",
        "界".repeat(260),
        "🙂".repeat(260),
        "文".repeat(180)
    );
    let scalar_segments = segment_text(&no_spaces, 255, scalar_pieces).unwrap();
    assert_eq!(scalar_segments.len(), 3);
    assert_eq!(reconstructed(&no_spaces, &scalar_segments), no_spaces);
    assert!(scalar_segments.iter().all(|segment| {
        segment.source_pieces <= 255
            && no_spaces.is_char_boundary(segment.content_range.start)
            && no_spaces.is_char_boundary(segment.content_range.end)
    }));
}

#[test]
fn leading_period_and_leading_newline_remain_lossless() {
    let text = format!("\r\n. Hidden start. {}", "x".repeat(300));
    let segments = segment_text(&text, 100, scalar_pieces).unwrap();

    assert_eq!(reconstructed(&text, &segments), text);
    assert!(segments.iter().all(|segment| segment.source_pieces <= 100));
}

#[test]
fn exact_piece_limit_boundaries_are_not_off_by_one() {
    for (pieces, expected_segments) in [(254, 1), (255, 1), (256, 2)] {
        let text = std::iter::repeat_n("x", pieces)
            .collect::<Vec<_>>()
            .join(" ");
        let segments = segment_text(&text, 255, word_pieces).unwrap();
        assert_eq!(segments.len(), expected_segments, "pieces={pieces}");
        assert!(segments.iter().all(|segment| segment.source_pieces <= 255));
        assert_eq!(reconstructed(&text, &segments), text);
    }
}

#[test]
fn whitespace_only_input_is_one_separator_only_segment() {
    let text = " \t\r\n\n\u{2003}";
    let segments = segment_text(text, 255, word_pieces).unwrap();

    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].content(text), "");
    assert_eq!(segments[0].separator(text), text);
    assert_eq!(segments[0].source_pieces, 0);
}

#[test]
fn rejects_zero_limit_encoder_failure_and_unsplitable_scalar() {
    assert_eq!(
        segment_text("text", 0, word_pieces).unwrap_err(),
        SegmentError::ZeroPieceLimit
    );

    let error = segment_text("bad", 3, |_| Err("broken model".into())).unwrap_err();
    assert!(matches!(
        error,
        SegmentError::Encoding {
            start: 0,
            end: 3,
            ..
        }
    ));

    let error = segment_text("🙂🙂", 1, |value| Ok(value.chars().count() * 2)).unwrap_err();
    assert!(matches!(
        error,
        SegmentError::UnsplitableScalar {
            start: 0,
            end: 4,
            pieces: 2,
            maximum: 1
        }
    ));
}

#[test]
fn enforces_input_and_segment_safety_limits() {
    let oversized = "x".repeat(segmenter::MAX_SEGMENTER_INPUT_BYTES + 1);
    assert!(matches!(
        segment_text(&oversized, 255, scalar_pieces).unwrap_err(),
        SegmentError::InputTooLarge { .. }
    ));

    let text = "x".repeat(segmenter::MAX_SEGMENTS + 1);
    let error = segment_text(&text, 1, scalar_pieces).unwrap_err();
    assert!(matches!(
        error,
        SegmentError::TooManySegments {
            maximum: segmenter::MAX_SEGMENTS
        } | SegmentError::EncodingWorkLimit { .. }
    ));
}

#[test]
#[ignore = "requires the converted en-zh SentencePiece model"]
fn real_sentencepiece_enforces_the_exact_limit() {
    let model_dir = std::env::var_os("MARIAN_CPU_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/enzh")
        });
    let tokenizer = marian_tokenizer::Tokenizer::open(model_dir.join("source.spm")).unwrap();
    let text = format!(
        "Dr. Smith reviewed version 1.2.3 at example.com. {} \r\n\r\n{}",
        "A long audit sentence with numbers 123.45 and emoji 🙂; ".repeat(180),
        "无空格中文🙂".repeat(300)
    );
    let segments = segment_text(&text, 255, |content| {
        tokenizer
            .encode(content)
            .map(|pieces| pieces.len())
            .map_err(|error| error.to_string())
    })
    .unwrap();

    assert_eq!(reconstructed(&text, &segments), text);
    assert!(segments.len() > 2);
    for segment in segments {
        let exact = tokenizer.encode(segment.content(&text)).unwrap().len();
        assert_eq!(segment.source_pieces, exact);
        assert!(exact <= 255);
    }
}
