//! A `miette::SourceCode` implementation with the same observable behavior
//! as scanning the whole document per span, but sub-linear in document
//! size: `friction check --format text` builds one [`FindingDiagnostic`]
//! (`crate::diagnostics`) per detected span and calls `render_report` on
//! each — and `miette`'s own built-in [`SourceCode`] impls for
//! `str`/`Arc<str>` (`context_info` in `miette::source_impls`) scan from
//! byte 0 of the WHOLE source up to every span's offset to find its
//! line/column and surrounding context lines. That makes one `check` run's
//! total rendering cost `O(spans * document_size)`: measured on a 10 MB,
//! 244-span document, `render_spans` alone took 3.2s of `check`'s ~11s
//! total: by far its most severe scaling defect (spans grew 8.4x from the 1
//! MB fixture, render time grew 83x).
//!
//! [`LineIndexedSource`] fixes this the safe way: it does not reimplement
//! `miette`'s line-scanning logic (`context_info`'s CRLF handling, sliding
//! "before" window, and post-span "after" trimming are exactly reproduced
//! below, unchanged — [`context_info`] is copied verbatim from
//! `miette-7.6.0/src/source_impls.rs`, MIT/Apache-2.0). The only new logic
//! is *where the scan starts*: instead of always starting at byte 0,
//! [`LineIndexedSource::read_span`] uses a once-built line-start index to
//! jump directly to a real line boundary at most `context_lines_before`
//! lines before the span, runs the unmodified scan over just that
//! sub-slice (bounded by a handful of lines, not the whole document), then
//! shifts the three fields the sub-slice scan reports relative to *its
//! own* start (`line`, `line_count`, `span.offset()`) back into
//! whole-document coordinates by adding the window's own starting line/
//! offset. `data` and `column` need no adjustment: `data` already borrows
//! from the original buffer (a sub-slice's bytes are the same bytes), and
//! `column` is already relative to whichever line it's anchored to,
//! independent of which window was scanned.
//!
//! # Why not just slice the document once, up front
//!
//! A window starting mid-document has no memory of how many lines came
//! before it, so a scan starting there would report `line 0` for
//! whatever line the window happens to start on. [`context_info`]'s
//! "before" window is populated by literally walking through it during
//! the scan (see the module docs on the sliding `before_lines_starts`
//! queue) — which is exactly why the sub-slice must start `context_lines_before`
//! real lines early: by the time the (still-unmodified) scan reaches the
//! span, that queue has filled the same way it would have from a
//! whole-document scan reaching the same point, so `line`/`span.offset()`
//! only need a constant `+= window_start_line` / `+= window_start_offset`
//! shift, not a full recomputation.

use std::sync::Arc;

use miette::{MietteError, MietteSpanContents, SourceCode, SourceSpan, SpanContents};

/// A `str`-backed [`SourceCode`] whose [`read_span`](SourceCode::read_span)
/// costs `O(context window size + log(line count))` instead of `str`'s own
/// `O(document size)` — see this module's own docs for why that's safe.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LineIndexedSource {
    source: Arc<str>,
    /// Byte offset of each line's first byte, `miette`'s own line-break
    /// convention (`\r\n`, lone `\r`, or lone `\n`, each counted once —
    /// see [`line_starts`]). Always begins with `0`.
    line_starts: Vec<usize>,
}

impl LineIndexedSource {
    /// Builds the line-start index once; every [`read_span`](SourceCode::read_span)
    /// call reuses it.
    #[must_use]
    pub fn new(source: Arc<str>) -> Self {
        let line_starts = line_starts(source.as_bytes());
        Self {
            source,
            line_starts,
        }
    }
}

/// Byte offset of every line start in `bytes`, `miette`'s own convention:
/// each `\r\n` pair, lone `\r`, or lone `\n` ends exactly one line (see
/// `context_info`'s `matches!(char, b'\r' | b'\n')` check and its
/// `iter.next_if_eq(&b'\n')` CRLF collapse below). Always starts with `0`.
fn line_starts(bytes: &[u8]) -> Vec<usize> {
    let mut starts = vec![0usize];
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'\r' => {
                i += 1;
                if bytes.get(i) == Some(&b'\n') {
                    i += 1;
                }
                starts.push(i);
            }
            b'\n' => {
                i += 1;
                starts.push(i);
            }
            _ => i += 1,
        }
    }
    starts
}

impl SourceCode for LineIndexedSource {
    fn read_span<'a>(
        &'a self,
        span: &SourceSpan,
        context_lines_before: usize,
        context_lines_after: usize,
    ) -> Result<Box<dyn SpanContents<'a> + 'a>, MietteError> {
        // The 0-indexed line containing `span`'s start: the last line
        // whose start offset is `<= span.offset()`.
        let target_line = self
            .line_starts
            .partition_point(|&start| start <= span.offset())
            .saturating_sub(1);
        let window_start_line = target_line.saturating_sub(context_lines_before);
        let window_start = self.line_starts[window_start_line];

        let window = &self.source.as_bytes()[window_start..];
        let window_span: SourceSpan = (span.offset() - window_start, span.len()).into();

        let contents = context_info(
            window,
            &window_span,
            context_lines_before,
            context_lines_after,
        )?;

        // `data` already borrows from `self.source` (a sub-slice of it)
        // and needs no adjustment. `column` is relative to whichever line
        // it's already anchored to, likewise unaffected by which window
        // was scanned. `line` and `line_count` both count newlines from
        // the scanned input's own start (`context_info`'s `line_count`
        // local var) — since that start is `window_start` here, not byte
        // 0 of the whole document, both need the same `+=
        // window_start_line` shift back into whole-document coordinates.
        Ok(Box::new(MietteSpanContents::new(
            contents.data(),
            (
                contents.span().offset() + window_start,
                contents.span().len(),
            )
                .into(),
            contents.line() + window_start_line,
            contents.column(),
            contents.line_count() + window_start_line,
        )))
    }
}

/// Verbatim copy of `miette::source_impls::context_info` (`miette-7.6.0`,
/// MIT/Apache-2.0) — the exact algorithm `str`'s own [`SourceCode`] impl
/// runs, unchanged, so [`LineIndexedSource::read_span`] only has to prove
/// its *window selection* is correct (see this module's own docs), never
/// this scan itself. Duplicated rather than called because the upstream
/// function is private to that crate.
///
/// `#[allow(clippy::collapsible_if)]`: kept byte-for-byte identical to the
/// vendored original rather than restructured to satisfy the lint:
/// deliberately, since this function's whole reason to exist is to be an
/// unmodified, independently-already-tested (upstream) reference the
/// windowed-scan tests above diff against.
#[allow(clippy::collapsible_if)]
fn context_info<'a>(
    input: &'a [u8],
    span: &SourceSpan,
    context_lines_before: usize,
    context_lines_after: usize,
) -> Result<MietteSpanContents<'a>, MietteError> {
    let mut offset = 0usize;
    let mut line_count = 0usize;
    let mut start_line = 0usize;
    let mut start_column = 0usize;
    let mut before_lines_starts = std::collections::VecDeque::new();
    let mut current_line_start = 0usize;
    let mut end_lines = 0usize;
    let mut post_span = false;
    let mut post_span_got_newline = false;
    let mut iter = input.iter().copied().peekable();
    while let Some(char) = iter.next() {
        if matches!(char, b'\r' | b'\n') {
            line_count += 1;
            if char == b'\r' && iter.next_if_eq(&b'\n').is_some() {
                offset += 1;
            }
            if offset < span.offset() {
                start_column = 0;
                before_lines_starts.push_back(current_line_start);
                if before_lines_starts.len() > context_lines_before {
                    start_line += 1;
                    before_lines_starts.pop_front();
                }
            } else if offset >= span.offset() + span.len().saturating_sub(1) {
                if post_span {
                    start_column = 0;
                    if post_span_got_newline {
                        end_lines += 1;
                    } else {
                        post_span_got_newline = true;
                    }
                    if end_lines >= context_lines_after {
                        offset += 1;
                        break;
                    }
                }
            }
            current_line_start = offset + 1;
        } else if offset < span.offset() {
            start_column += 1;
        }

        if offset >= (span.offset() + span.len()).saturating_sub(1) {
            post_span = true;
            if end_lines >= context_lines_after {
                offset += 1;
                break;
            }
        }

        offset += 1;
    }

    if offset >= (span.offset() + span.len()).saturating_sub(1) {
        let starting_offset = before_lines_starts.front().copied().unwrap_or_else(|| {
            if context_lines_before == 0 {
                span.offset()
            } else {
                0
            }
        });
        Ok(MietteSpanContents::new(
            &input[starting_offset..offset],
            (starting_offset, offset - starting_offset).into(),
            start_line,
            if context_lines_before == 0 {
                start_column
            } else {
                0
            },
            line_count,
        ))
    } else {
        Err(MietteError::OutOfBounds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs `str`'s own (whole-document-scanning) `SourceCode` impl and
    /// [`LineIndexedSource`]'s windowed one over the same `(source, span,
    /// context_lines_before, context_lines_after)` and asserts every
    /// [`SpanContents`] field agrees: the correctness oracle for the
    /// window-selection logic this module adds. `str`'s own impl is
    /// exactly [`context_info`] called with `window_start == 0`, so this
    /// is a direct differential test against the unmodified algorithm.
    fn assert_matches_whole_document_scan(
        source: &str,
        span: SourceSpan,
        context_lines_before: usize,
        context_lines_after: usize,
    ) {
        let expected = source
            .read_span(&span, context_lines_before, context_lines_after)
            .expect("str's own SourceCode impl must resolve this span");
        let indexed = LineIndexedSource::new(Arc::from(source));
        let actual = indexed
            .read_span(&span, context_lines_before, context_lines_after)
            .expect("LineIndexedSource must resolve the same span");

        assert_eq!(actual.data(), expected.data(), "data (source={source:?})");
        assert_eq!(actual.span(), expected.span(), "span (source={source:?})");
        assert_eq!(actual.line(), expected.line(), "line (source={source:?})");
        assert_eq!(
            actual.column(),
            expected.column(),
            "column (source={source:?})"
        );
        assert_eq!(
            actual.line_count(),
            expected.line_count(),
            "line_count (source={source:?})"
        );
    }

    const SAMPLE: &str = "one two three\nfour five six\nseven eight nine\nten eleven twelve\nthirteen fourteen fifteen\n";

    #[test]
    fn matches_whole_document_scan_for_a_mid_document_span() {
        // "seven" on line 2 (0-indexed).
        let start = SAMPLE.find("seven").unwrap();
        assert_matches_whole_document_scan(SAMPLE, (start, 5).into(), 1, 1);
    }

    #[test]
    fn matches_whole_document_scan_with_zero_context_lines() {
        let start = SAMPLE.find("eleven").unwrap();
        assert_matches_whole_document_scan(SAMPLE, (start, 6).into(), 0, 0);
    }

    #[test]
    fn matches_whole_document_scan_with_more_context_than_lines_available() {
        // Only 2 lines precede this span's line; asking for 10 must clamp,
        // not panic or diverge from the whole-document scan's own clamp.
        let start = SAMPLE.find("seven").unwrap();
        assert_matches_whole_document_scan(SAMPLE, (start, 5).into(), 10, 10);
    }

    #[test]
    fn matches_whole_document_scan_for_a_span_at_the_very_start() {
        assert_matches_whole_document_scan(SAMPLE, (0, 3).into(), 2, 2);
    }

    #[test]
    fn matches_whole_document_scan_for_a_span_at_the_very_end() {
        let start = SAMPLE.rfind("fifteen").unwrap();
        assert_matches_whole_document_scan(SAMPLE, (start, 7).into(), 2, 2);
    }

    #[test]
    fn matches_whole_document_scan_for_a_span_spanning_multiple_lines() {
        let start = SAMPLE.find("six").unwrap();
        let end = SAMPLE.find("eight").unwrap();
        assert_matches_whole_document_scan(SAMPLE, (start, end - start).into(), 1, 1);
    }

    #[test]
    fn matches_whole_document_scan_for_a_zero_width_span() {
        let start = SAMPLE.find("nine").unwrap();
        assert_matches_whole_document_scan(SAMPLE, (start, 0).into(), 1, 1);
    }

    #[test]
    fn matches_whole_document_scan_with_crlf_line_endings() {
        let crlf = SAMPLE.replace('\n', "\r\n");
        let start = crlf.find("seven").unwrap();
        assert_matches_whole_document_scan(&crlf, (start, 5).into(), 1, 1);
    }

    #[test]
    fn matches_whole_document_scan_with_lone_cr_line_endings() {
        let cr = SAMPLE.replace('\n', "\r");
        let start = cr.find("seven").unwrap();
        assert_matches_whole_document_scan(&cr, (start, 5).into(), 1, 1);
    }

    #[test]
    fn matches_whole_document_scan_for_a_single_line_document() {
        assert_matches_whole_document_scan("no newlines here at all", (3, 9).into(), 1, 1);
    }

    #[test]
    fn matches_whole_document_scan_for_every_span_over_a_realistic_sample() {
        // Every word in SAMPLE, every (before, after) pair in 0..=3: a
        // broad sweep rather than a handful of hand-picked offsets.
        for (start, _) in SAMPLE.char_indices() {
            for before in 0..=3 {
                for after in 0..=3 {
                    assert_matches_whole_document_scan(SAMPLE, (start, 1).into(), before, after);
                }
            }
        }
    }

    #[test]
    fn line_starts_always_begins_with_zero() {
        assert_eq!(line_starts(b""), vec![0]);
        assert_eq!(line_starts(b"no newlines"), vec![0]);
    }

    #[test]
    fn line_starts_collapses_crlf_into_one_break() {
        assert_eq!(line_starts(b"a\r\nb"), vec![0, 3]);
    }

    #[test]
    fn line_starts_treats_lone_cr_and_lone_lf_as_breaks() {
        assert_eq!(line_starts(b"a\rb\nc"), vec![0, 2, 4]);
    }
}
