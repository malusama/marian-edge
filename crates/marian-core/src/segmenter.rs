//! Bounded, tokenizer-aware segmentation for long source text.
//!
//! The splitter deliberately has no tokenizer dependency. Callers provide an
//! exact piece-counting closure and append each [`TextSegment::separator`] to
//! the translated content verbatim. This keeps paragraph formatting outside
//! SentencePiece normalization while guaranteeing that every non-empty
//! content range fits the caller's source-piece limit.

use std::collections::HashMap;
use std::fmt;
use std::ops::Range;

/// Maximum accepted input size for one segmentation operation.
pub const MAX_SEGMENTER_INPUT_BYTES: usize = 4 * 1024 * 1024;
/// Maximum number of ranges produced for one input.
pub const MAX_SEGMENTS: usize = 4_096;
/// Maximum tokenizer invocations, including sizing probes.
pub const MAX_ENCODING_CALLS: usize = 32_768;
/// Maximum cumulative bytes passed to the tokenizer during sizing probes.
pub const MAX_ENCODING_BYTES: usize = 64 * 1024 * 1024;
/// Maximum cumulative pieces observed during sizing probes.
pub const MAX_COUNTED_PIECES: usize = 8 * 1024 * 1024;

/// A source fragment to translate followed by source text to preserve exactly.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TextSegment {
    pub content_range: Range<usize>,
    pub separator_range: Range<usize>,
    pub source_pieces: usize,
}

impl TextSegment {
    pub fn content<'a>(&self, source: &'a str) -> &'a str {
        &source[self.content_range.clone()]
    }

    pub fn separator<'a>(&self, source: &'a str) -> &'a str {
        &source[self.separator_range.clone()]
    }

    pub fn full_range(&self) -> Range<usize> {
        self.content_range.start..self.separator_range.end
    }
}

/// Failure returned before unbounded tokenizer or segmentation work can occur.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SegmentError {
    ZeroPieceLimit,
    InputTooLarge {
        bytes: usize,
        maximum: usize,
    },
    Encoding {
        start: usize,
        end: usize,
        message: String,
    },
    EncodingWorkLimit {
        resource: &'static str,
        maximum: usize,
    },
    TooManySegments {
        maximum: usize,
    },
    UnsplitableScalar {
        start: usize,
        end: usize,
        pieces: usize,
        maximum: usize,
    },
}

impl fmt::Display for SegmentError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroPieceLimit => formatter.write_str("source piece limit must be non-zero"),
            Self::InputTooLarge { bytes, maximum } => write!(
                formatter,
                "source contains {bytes} bytes; segmenter maximum is {maximum}"
            ),
            Self::Encoding {
                start,
                end,
                message,
            } => write!(
                formatter,
                "source tokenizer failed for byte range {start}..{end}: {message}"
            ),
            Self::EncodingWorkLimit { resource, maximum } => write!(
                formatter,
                "source segmentation exceeded {resource} safety limit {maximum}"
            ),
            Self::TooManySegments { maximum } => {
                write!(formatter, "source requires more than {maximum} segments")
            }
            Self::UnsplitableScalar {
                start,
                end,
                pieces,
                maximum,
            } => write!(
                formatter,
                "UTF-8 scalar at byte range {start}..{end} encodes to {pieces} pieces; maximum is {maximum}"
            ),
        }
    }
}

impl std::error::Error for SegmentError {}

/// Splits `text` into ordered, lossless ranges whose content fits `max_pieces`.
///
/// `count_pieces` must return the exact number of source pieces for its input.
/// Its errors are converted to [`SegmentError::Encoding`]. The returned ranges
/// partition `text` without overlap or loss. Separators are never passed to the
/// translation model and should be appended to translated content verbatim.
///
/// A single sentence that already fits is returned unchanged. Multiple
/// sentences and line breaks remain separate even when their combined text
/// fits, so model normalization cannot erase the original separators.
pub fn segment_text<F>(
    text: &str,
    max_pieces: usize,
    mut count_pieces: F,
) -> Result<Vec<TextSegment>, SegmentError>
where
    F: FnMut(&str) -> Result<usize, String>,
{
    if max_pieces == 0 {
        return Err(SegmentError::ZeroPieceLimit);
    }
    if text.len() > MAX_SEGMENTER_INPUT_BYTES {
        return Err(SegmentError::InputTooLarge {
            bytes: text.len(),
            maximum: MAX_SEGMENTER_INPUT_BYTES,
        });
    }

    let content_end = trim_trailing_whitespace(text, text.len());
    let mut counter = PieceCounter::new(text, &mut count_pieces);
    let whole_pieces = counter.count(0..content_end)?;
    let units = sentence_units(text, content_end);
    if whole_pieces <= max_pieces && units.len() <= 1 {
        return Ok(vec![TextSegment {
            content_range: 0..content_end,
            separator_range: content_end..text.len(),
            source_pieces: whole_pieces,
        }]);
    }

    let mut result = Vec::new();
    for unit in units {
        let pieces = counter.count(unit.content.clone())?;
        if pieces <= max_pieces {
            push_segment(
                &mut result,
                TextSegment {
                    content_range: unit.content,
                    separator_range: unit.separator,
                    source_pieces: pieces,
                },
            )?;
        } else {
            split_oversize_unit(text, unit, max_pieces, &mut counter, &mut result)?;
        }
    }

    debug_assert_lossless(text, &result);
    Ok(result)
}

#[derive(Clone, Debug)]
struct Unit {
    content: Range<usize>,
    separator: Range<usize>,
}

#[derive(Clone, Copy, Debug)]
struct BreakPoint {
    content_end: usize,
    next_start: usize,
    priority: u8,
}

struct PieceCounter<'text, 'closure, F> {
    text: &'text str,
    closure: &'closure mut F,
    cache: HashMap<(usize, usize), usize>,
    calls: usize,
    bytes: usize,
    pieces: usize,
}

impl<'text, 'closure, F> PieceCounter<'text, 'closure, F>
where
    F: FnMut(&str) -> Result<usize, String>,
{
    fn new(text: &'text str, closure: &'closure mut F) -> Self {
        Self {
            text,
            closure,
            cache: HashMap::new(),
            calls: 0,
            bytes: 0,
            pieces: 0,
        }
    }

    fn count(&mut self, range: Range<usize>) -> Result<usize, SegmentError> {
        if let Some(&pieces) = self.cache.get(&(range.start, range.end)) {
            return Ok(pieces);
        }

        self.calls = self
            .calls
            .checked_add(1)
            .ok_or(SegmentError::EncodingWorkLimit {
                resource: "tokenizer calls",
                maximum: MAX_ENCODING_CALLS,
            })?;
        if self.calls > MAX_ENCODING_CALLS {
            return Err(SegmentError::EncodingWorkLimit {
                resource: "tokenizer calls",
                maximum: MAX_ENCODING_CALLS,
            });
        }
        self.bytes =
            self.bytes
                .checked_add(range.len())
                .ok_or(SegmentError::EncodingWorkLimit {
                    resource: "encoded bytes",
                    maximum: MAX_ENCODING_BYTES,
                })?;
        if self.bytes > MAX_ENCODING_BYTES {
            return Err(SegmentError::EncodingWorkLimit {
                resource: "encoded bytes",
                maximum: MAX_ENCODING_BYTES,
            });
        }

        let pieces = (self.closure)(&self.text[range.clone()]).map_err(|message| {
            SegmentError::Encoding {
                start: range.start,
                end: range.end,
                message,
            }
        })?;
        self.pieces = self
            .pieces
            .checked_add(pieces)
            .ok_or(SegmentError::EncodingWorkLimit {
                resource: "counted pieces",
                maximum: MAX_COUNTED_PIECES,
            })?;
        if self.pieces > MAX_COUNTED_PIECES {
            return Err(SegmentError::EncodingWorkLimit {
                resource: "counted pieces",
                maximum: MAX_COUNTED_PIECES,
            });
        }
        self.cache.insert((range.start, range.end), pieces);
        Ok(pieces)
    }
}

fn sentence_units(text: &str, content_end: usize) -> Vec<Unit> {
    let mut units = Vec::new();
    let mut cursor = 0;
    let mut index = 0;

    while index < content_end {
        let (character, next) = next_scalar(text, index);
        if is_terminator(character) {
            let punctuation_end = consume_terminators(text, index, content_end);
            let quoted_end = consume_closing_marks(text, punctuation_end, content_end);
            if boundary_follows(text, quoted_end, content_end)
                && (character != '.' || period_can_end_sentence(text, index, quoted_end))
            {
                let separator_end = consume_whitespace(text, quoted_end, content_end);
                units.push(Unit {
                    content: cursor..quoted_end,
                    separator: quoted_end..separator_end,
                });
                cursor = separator_end;
                index = separator_end;
                continue;
            }
            index = punctuation_end.max(next);
            continue;
        }

        if character.is_whitespace() {
            let whitespace_end = consume_whitespace(text, index, content_end);
            if text[index..whitespace_end]
                .chars()
                .any(|item| item == '\n' || item == '\r')
            {
                units.push(Unit {
                    content: cursor..index,
                    separator: index..whitespace_end,
                });
                cursor = whitespace_end;
            }
            index = whitespace_end;
            continue;
        }
        index = next;
    }

    if cursor < content_end {
        units.push(Unit {
            content: cursor..content_end,
            separator: content_end..text.len(),
        });
    } else if let Some(last) = units.last_mut() {
        last.separator.end = text.len();
    }
    units
}

fn split_oversize_unit<F>(
    text: &str,
    unit: Unit,
    max_pieces: usize,
    counter: &mut PieceCounter<'_, '_, F>,
    result: &mut Vec<TextSegment>,
) -> Result<(), SegmentError>
where
    F: FnMut(&str) -> Result<usize, String>,
{
    let break_points = fallback_break_points(text, unit.content.clone());
    let mut start = unit.content.start;

    while start < unit.content.end {
        let remaining = start..unit.content.end;
        let remaining_pieces = counter.count(remaining.clone())?;
        if remaining_pieces <= max_pieces {
            push_segment(
                result,
                TextSegment {
                    content_range: remaining,
                    separator_range: unit.separator.clone(),
                    source_pieces: remaining_pieces,
                },
            )?;
            return Ok(());
        }

        let mut selected = None;
        for priority in 0..=2 {
            if let Some(candidate) = rightmost_fitting_break(
                &break_points,
                priority,
                start,
                unit.content.end,
                max_pieces,
                counter,
            )? {
                selected = Some(candidate);
                break;
            }
        }

        let (content_end, next_start, pieces) = if let Some((point, pieces)) = selected {
            (point.content_end, point.next_start, pieces)
        } else {
            let (end, pieces) =
                largest_fitting_scalar_prefix(text, start, unit.content.end, max_pieces, counter)?;
            (end, end, pieces)
        };

        push_segment(
            result,
            TextSegment {
                content_range: start..content_end,
                separator_range: content_end..next_start,
                source_pieces: pieces,
            },
        )?;
        start = next_start;
    }

    if unit.separator.start < unit.separator.end {
        if let Some(last) = result.last_mut() {
            last.separator_range.end = unit.separator.end;
        }
    }
    Ok(())
}

fn rightmost_fitting_break<F>(
    points: &[BreakPoint],
    priority: u8,
    start: usize,
    end: usize,
    max_pieces: usize,
    counter: &mut PieceCounter<'_, '_, F>,
) -> Result<Option<(BreakPoint, usize)>, SegmentError>
where
    F: FnMut(&str) -> Result<usize, String>,
{
    let candidates = points
        .iter()
        .copied()
        .filter(|point| {
            point.priority == priority
                && point.content_end > start
                && point.next_start > start
                && point.next_start < end
        })
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Ok(None);
    }

    let mut low = 0;
    let mut high = candidates.len();
    let mut best = None;
    while low < high {
        let middle = low + (high - low) / 2;
        let candidate = candidates[middle];
        let pieces = counter.count(start..candidate.content_end)?;
        if pieces <= max_pieces {
            best = Some((candidate, pieces));
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    Ok(best)
}

fn largest_fitting_scalar_prefix<F>(
    text: &str,
    start: usize,
    end: usize,
    max_pieces: usize,
    counter: &mut PieceCounter<'_, '_, F>,
) -> Result<(usize, usize), SegmentError>
where
    F: FnMut(&str) -> Result<usize, String>,
{
    let first_end = next_scalar(text, start).1;
    let first_pieces = counter.count(start..first_end)?;
    if first_pieces > max_pieces {
        return Err(SegmentError::UnsplitableScalar {
            start,
            end: first_end,
            pieces: first_pieces,
            maximum: max_pieces,
        });
    }

    let mut low = first_end;
    let mut high = end;
    let mut best = (first_end, first_pieces);
    while low < high {
        let raw_middle = low + (high - low).div_ceil(2);
        let middle = floor_char_boundary(text, raw_middle, start);
        if middle <= best.0 {
            break;
        }
        let pieces = counter.count(start..middle)?;
        if pieces <= max_pieces {
            best = (middle, pieces);
            low = middle;
        } else {
            high = previous_char_boundary(text, middle, start);
        }
    }
    Ok(best)
}

fn fallback_break_points(text: &str, range: Range<usize>) -> Vec<BreakPoint> {
    let mut points = Vec::new();
    let mut index = range.start;
    while index < range.end {
        let (character, next) = next_scalar(text, index);
        if character.is_whitespace() {
            let end = consume_whitespace(text, index, range.end);
            points.push(BreakPoint {
                content_end: index,
                next_start: end,
                priority: 2,
            });
            index = end;
            continue;
        }

        let priority = match character {
            ';' | '；' | ':' | '：' | '/' | '\u{2013}' | '\u{2014}' => Some(0),
            ',' | '，' => Some(1),
            '-' if is_dash_boundary(text, index, range.clone()) => Some(0),
            _ => None,
        };
        if let Some(priority) = priority {
            let mut punctuation_end = next;
            if character == '-' {
                while punctuation_end < range.end && text.as_bytes()[punctuation_end] == b'-' {
                    punctuation_end += 1;
                }
            }
            let separator_end = consume_whitespace(text, punctuation_end, range.end);
            points.push(BreakPoint {
                content_end: punctuation_end,
                next_start: separator_end,
                priority,
            });
            index = punctuation_end;
            continue;
        }
        index = next;
    }
    points
}

fn push_segment(result: &mut Vec<TextSegment>, segment: TextSegment) -> Result<(), SegmentError> {
    if result.len() >= MAX_SEGMENTS {
        return Err(SegmentError::TooManySegments {
            maximum: MAX_SEGMENTS,
        });
    }
    result.push(segment);
    Ok(())
}

fn trim_trailing_whitespace(text: &str, mut end: usize) -> usize {
    while end > 0 {
        let (start, character) = previous_scalar(text, end);
        if !character.is_whitespace() {
            break;
        }
        end = start;
    }
    end
}

fn consume_terminators(text: &str, mut index: usize, end: usize) -> usize {
    while index < end {
        let (character, next) = next_scalar(text, index);
        if !is_terminator(character) {
            break;
        }
        index = next;
    }
    index
}

fn consume_closing_marks(text: &str, mut index: usize, end: usize) -> usize {
    while index < end {
        let (character, next) = next_scalar(text, index);
        if !matches!(
            character,
            '"' | '\''
                | '\u{2019}'
                | '\u{201d}'
                | '\u{00bb}'
                | ')'
                | ']'
                | '}'
                | '\u{3009}'
                | '\u{300b}'
                | '\u{300d}'
                | '\u{300f}'
                | '\u{3011}'
                | '\u{3015}'
                | '\u{3017}'
                | '\u{3019}'
                | '\u{301b}'
        ) {
            break;
        }
        index = next;
    }
    index
}

fn consume_whitespace(text: &str, mut index: usize, end: usize) -> usize {
    while index < end {
        let (character, next) = next_scalar(text, index);
        if !character.is_whitespace() {
            break;
        }
        index = next;
    }
    index
}

fn boundary_follows(text: &str, index: usize, end: usize) -> bool {
    index == end || (index < end && next_scalar(text, index).0.is_whitespace())
}

fn is_terminator(character: char) -> bool {
    matches!(
        character,
        '.' | '!' | '?' | '\u{3002}' | '\u{ff01}' | '\u{ff1f}' | '\u{2026}'
    )
}

fn period_can_end_sentence(text: &str, index: usize, quoted_end: usize) -> bool {
    let previous = (index > 0).then(|| previous_scalar(text, index).1);
    let next = if index + 1 < text.len() {
        Some(next_scalar(text, index + 1).0)
    } else {
        None
    };
    if previous.is_some_and(|item| item.is_ascii_digit())
        && next.is_some_and(|item| item.is_ascii_digit())
    {
        return false;
    }

    let token_start = text[..index]
        .char_indices()
        .rev()
        .find_map(|(offset, character)| {
            (character.is_whitespace() || matches!(character, '(' | '[' | '{' | '"' | '\u{201c}'))
                .then_some(offset + character.len_utf8())
        })
        .unwrap_or(0);
    let mut token = text[token_start..=index].to_ascii_lowercase();
    token.retain(|character| character.is_ascii_alphanumeric() || character == '.');

    let following = text[quoted_end..].trim_start_matches(char::is_whitespace);
    let following_character = following.chars().next();
    if TITLE_ABBREVIATIONS.contains(&token.as_str()) && following_character.is_some() {
        return false;
    }
    if ALWAYS_NON_TERMINAL_ABBREVIATIONS.contains(&token.as_str()) && following_character.is_some()
    {
        return false;
    }
    if COMMON_ABBREVIATIONS.contains(&token.as_str())
        && following_character
            .is_some_and(|item| item.is_ascii_lowercase() || item.is_ascii_digit())
    {
        return false;
    }
    if is_initial_or_acronym(&token)
        && following_character.is_some_and(|item| item.is_ascii_uppercase())
    {
        return false;
    }
    true
}

const TITLE_ABBREVIATIONS: &[&str] = &[
    "capt.", "col.", "dr.", "gen.", "gov.", "lt.", "mr.", "mrs.", "ms.", "prof.", "rep.", "rev.",
    "sen.", "sgt.", "sr.", "st.",
];

const ALWAYS_NON_TERMINAL_ABBREVIATIONS: &[&str] = &[
    "e.g.", "i.e.", "fig.", "no.", "vs.", "jan.", "feb.", "mar.", "apr.", "jun.", "jul.", "aug.",
    "sep.", "sept.", "oct.", "nov.", "dec.",
];

const COMMON_ABBREVIATIONS: &[&str] = &[
    "approx.", "co.", "corp.", "dept.", "ed.", "eds.", "est.", "etc.", "inc.", "ltd.", "misc.",
    "p.", "pp.", "vol.",
];

fn is_initial_or_acronym(token: &str) -> bool {
    let components = token.trim_end_matches('.').split('.').collect::<Vec<_>>();
    !components.is_empty()
        && components.len() <= 8
        && components
            .iter()
            .all(|component| component.len() == 1 && component.as_bytes()[0].is_ascii_alphabetic())
}

fn is_dash_boundary(text: &str, index: usize, range: Range<usize>) -> bool {
    let bytes = text.as_bytes();
    let repeated = index + 1 < range.end && bytes[index + 1] == b'-';
    let before_space = index == range.start || previous_scalar(text, index).1.is_whitespace();
    let after = index + 1;
    let after_space = after >= range.end || next_scalar(text, after).0.is_whitespace();
    repeated || before_space || after_space
}

fn next_scalar(text: &str, index: usize) -> (char, usize) {
    let character = text[index..]
        .chars()
        .next()
        .expect("index must precede text end");
    (character, index + character.len_utf8())
}

fn previous_scalar(text: &str, end: usize) -> (usize, char) {
    let character = text[..end]
        .chars()
        .next_back()
        .expect("end must follow a scalar");
    (end - character.len_utf8(), character)
}

fn floor_char_boundary(text: &str, mut index: usize, minimum: usize) -> usize {
    while index > minimum && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn previous_char_boundary(text: &str, index: usize, minimum: usize) -> usize {
    if index <= minimum {
        return minimum;
    }
    let mut previous = index - 1;
    while previous > minimum && !text.is_char_boundary(previous) {
        previous -= 1;
    }
    previous
}

fn debug_assert_lossless(_text: &str, _segments: &[TextSegment]) {
    #[cfg(debug_assertions)]
    {
        let mut cursor = 0;
        for segment in _segments {
            debug_assert_eq!(segment.content_range.start, cursor);
            debug_assert_eq!(segment.content_range.end, segment.separator_range.start);
            debug_assert!(segment.separator_range.end <= _text.len());
            cursor = segment.separator_range.end;
        }
        debug_assert_eq!(cursor, _text.len());
    }
}
