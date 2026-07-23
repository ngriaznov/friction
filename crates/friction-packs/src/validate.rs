//! Build-time pack audits over an [`InventoryPack`]: disjointness,
//! closure, output-frequency hygiene, and guard-token exposure.
//!
//! Distinct from and consistent with `friction-harness::closure`'s
//! existing *per-input runtime* check (which licenses `tokenize(replacement)`
//! wholesale for a hit that actually fired against real input text) — rule
//! 2 here vets the pack's replacements up front, at build time, which is
//! exactly what makes the runtime check trustworthy rather than vacuous.

use std::collections::BTreeSet;
use std::fmt;
use std::sync::LazyLock;

use regex::Regex;

use crate::inventory::InventoryPack;

/// `[a-z']+` — the same word-token shape `friction-harness::clean`'s own
/// tokenizer uses, reimplemented locally so this crate never has to
/// depend on `friction-harness` (which depends on this crate, not the
/// other way around).
static WORD_TOKEN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[a-z']+").expect("word-token pattern is valid"));

/// One pack-audit finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Violation {
    Disjointness(DisjointnessViolation),
    Closure(ClosureViolation),
    FrequencyHygiene(FrequencyHygieneViolation),
    GuardTokenExposure(GuardTokenExposureViolation),
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Disjointness(v) => v.fmt(f),
            Self::Closure(v) => v.fmt(f),
            Self::FrequencyHygiene(v) => v.fmt(f),
            Self::GuardTokenExposure(v) => v.fmt(f),
        }
    }
}

/// A `substitution_pairs` entry whose `replacement` string matches
/// another (or its own) candidate pattern — rule 1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisjointnessViolation {
    pub pair_id: Box<str>,
    pub replacement: Box<str>,
    pub conflicting_family: &'static str,
    pub conflicting_id: Box<str>,
}

impl fmt::Display for DisjointnessViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "disjointness: {}'s replacement {:?} matches {}.{}",
            self.pair_id, self.replacement, self.conflicting_family, self.conflicting_id
        )
    }
}

/// A `substitution_pairs` entry with an unattested content word — rule 2.
///
/// Its `replacement` contains a content word not traceable to the
/// pattern, `declared_literal_tokens`, the closure function-word
/// allowance, or an attested (and honestly noted) token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClosureViolation {
    pub pair_id: Box<str>,
    /// Sorted.
    pub offending_tokens: Vec<Box<str>>,
}

impl fmt::Display for ClosureViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "closure: {}'s replacement has unattested token(s): {}",
            self.pair_id,
            self.offending_tokens.join(", ")
        )
    }
}

/// Why an [`OutputFrequencyBand`](crate::OutputFrequencyBand) failed
/// hygiene — rule 3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrequencyHygieneReason {
    NoFiniteNonNegativeMax,
    ZeroSampleDocCount,
    MaxBelowObservedRate,
}

impl fmt::Display for FrequencyHygieneReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NoFiniteNonNegativeMax => "max is not finite and non-negative",
            Self::ZeroSampleDocCount => "sample_doc_count is zero",
            Self::MaxBelowObservedRate => "max is below the band's own observed_rate",
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrequencyHygieneViolation {
    pub frame: Box<str>,
    pub reason: FrequencyHygieneReason,
}

impl fmt::Display for FrequencyHygieneViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "frequency-hygiene: {:?}: {}", self.frame, self.reason)
    }
}

/// A `deletion_spans`/`substitution_pairs`/`ritual_frames`/`preview_frames`
/// entry whose entire literal-token content is drawn from
/// `guard_tokens` (negation/quantifiers/modals/connectives) — rule 4.
///
/// A pattern that reduces to nothing but a guard token (e.g. a bare
/// `\bnever\b`) edits a closed syntactic class the rest of the pack
/// treats as meaning-bearing and never touches; shipping one un-flagged
/// would let a negation/modal/quantifier/connective flip slip through
/// disguised as an ordinary lexical edit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuardTokenExposureViolation {
    pub family: &'static str,
    pub id: Box<str>,
    /// Sorted.
    pub guard_tokens: Vec<Box<str>>,
}

impl fmt::Display for GuardTokenExposureViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "guard-token-exposure: {}.{}'s pattern is composed entirely of guard token(s): {}",
            self.family,
            self.id,
            self.guard_tokens.join(", ")
        )
    }
}

/// Runs every rule against `pack`, in fixed order: disjointness, closure,
/// frequency hygiene, guard-token exposure.
#[must_use]
pub fn validate(pack: &InventoryPack) -> Vec<Violation> {
    let mut violations: Vec<Violation> = check_disjointness(pack)
        .into_iter()
        .map(Violation::Disjointness)
        .collect();
    violations.extend(check_closure(pack).into_iter().map(Violation::Closure));
    violations.extend(
        check_frequency_hygiene(pack)
            .into_iter()
            .map(Violation::FrequencyHygiene),
    );
    violations.extend(
        check_guard_token_exposure(pack)
            .into_iter()
            .map(Violation::GuardTokenExposure),
    );
    violations
}

/// Rule 1 — disjointness.
///
/// No `substitution_pairs` replacement may match any
/// `deletion_spans`/`substitution_pairs`/`ritual_frames`/`preview_frames`
/// pattern (including its own), pairwise and deterministically.
#[must_use]
pub fn check_disjointness(pack: &InventoryPack) -> Vec<DisjointnessViolation> {
    let mut candidates: Vec<(u8, &'static str, &str, &Regex)> = Vec::new();
    for span in pack.deletion_spans() {
        candidates.push((0, "deletion_spans", &span.id, &span.pattern));
    }
    for sub in pack.substitution_pairs() {
        candidates.push((1, "substitution_pairs", &sub.id, &sub.pattern));
    }
    for ritual in pack.ritual_frames() {
        candidates.push((2, "ritual_frames", &ritual.id, &ritual.pattern));
    }
    for preview in pack.preview_frames() {
        candidates.push((3, "preview_frames", &preview.id, &preview.pattern));
    }
    candidates.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.2.cmp(b.2)));

    let mut violations = Vec::new();
    for pair in pack.substitution_pairs() {
        for &(_, family, id, pattern) in &candidates {
            if replacement_matches_embedded(pattern, &pair.replacement) {
                violations.push(DisjointnessViolation {
                    pair_id: pair.id.clone(),
                    replacement: pair.replacement.clone(),
                    conflicting_family: family,
                    conflicting_id: Box::from(id),
                });
            }
        }
    }
    violations
}

/// Whether `pattern` would match `replacement` once it lands back in
/// running text, not just in bare isolation.
///
/// A bare `pattern.is_match(replacement)` misses patterns that require a
/// boundary-consuming whitespace character the bare replacement string
/// can't itself supply — a leading anchor like `^by following these
/// steps,?\s+` needs a trailing `\s`, a trailing anchor like `\s+to suit
/// your needs\b` needs a leading one. Once that replacement is spliced
/// into a real sentence, the surrounding whitespace *is* there and the
/// pattern fires. So this checks the bare string plus both single-space
/// paddings (trailing-only, to satisfy a leading `^...\s+` anchor;
/// leading-only, to satisfy a trailing `\s+...$`/`\b` anchor) and unions
/// the results.
fn replacement_matches_embedded(pattern: &Regex, replacement: &str) -> bool {
    pattern.is_match(replacement)
        || pattern.is_match(&format!("{replacement} "))
        || pattern.is_match(&format!(" {replacement}"))
}

/// Strips the small, fixed regex-syntax vocabulary this pack's authors
/// actually use (`(?i)`, `\b`, `^`, `$`, backslash-escapes, character
/// classes/quantifiers), lowercases, and re-tokenizes into `[a-z']+`
/// word tokens.
///
/// Best-effort by design: `declared_literal_tokens` is the escape hatch
/// for patterns this heuristic can't parse cleanly (e.g. alternation).
fn literal_tokens_from_pattern(pattern_source: &str) -> BTreeSet<String> {
    let without_flag = pattern_source.replace("(?i)", " ");
    let mut buf = String::with_capacity(without_flag.len());
    let mut chars = without_flag.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                // Drop the escape entirely: the backslash and whatever it
                // escapes (`\s`, `\b`, `\.`, ...) is regex syntax, never
                // literal text.
                chars.next();
            }
            '^' | '$' | '.' | '*' | '+' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' => {}
            other => buf.push(other),
        }
    }
    let lower = buf.to_lowercase();
    WORD_TOKEN
        .find_iter(&lower)
        .map(|m| m.as_str().to_string())
        .collect()
}

/// Rule 2 — closure.
///
/// Every content-word token in a `substitution_pairs` entry's
/// `replacement` must be traceable to its own pattern (heuristically),
/// `declared_literal_tokens`, the pack's `closure_function_word_allowance`,
/// or `attested_replacement_tokens` (only when `notes` is non-empty — the
/// honesty requirement).
#[must_use]
pub fn check_closure(pack: &InventoryPack) -> Vec<ClosureViolation> {
    let function_words: BTreeSet<String> = pack
        .closure_function_word_allowance()
        .iter()
        .map(|t| t.to_lowercase())
        .collect();

    let mut violations = Vec::new();
    for pair in pack.substitution_pairs() {
        let mut allowed = literal_tokens_from_pattern(&pair.pattern_source);
        for t in &pair.declared_literal_tokens {
            allowed.insert(t.to_lowercase());
        }
        allowed.extend(function_words.iter().cloned());

        let notes_non_empty = pair.notes.as_deref().is_some_and(|n| !n.trim().is_empty());
        let attested: BTreeSet<String> = if notes_non_empty {
            pair.attested_replacement_tokens
                .iter()
                .map(|t| t.to_lowercase())
                .collect()
        } else {
            BTreeSet::new()
        };

        let replacement_lower = pair.replacement.to_lowercase();
        let mut offending: Vec<Box<str>> = WORD_TOKEN
            .find_iter(&replacement_lower)
            .map(|m| m.as_str().to_string())
            .filter(|token| !allowed.contains(token) && !attested.contains(token))
            .map(String::into_boxed_str)
            .collect();
        offending.sort();
        offending.dedup();

        if !offending.is_empty() {
            violations.push(ClosureViolation {
                pair_id: pair.id.clone(),
                offending_tokens: offending,
            });
        }
    }
    violations
}

/// Rule 3 — output-frequency hygiene.
///
/// For every `substitution_pairs` replacement that a band names
/// (case-insensitively), the band's `max` must be finite, non-negative,
/// based on at least one sample doc, and no lower than the band's own
/// `observed_rate`. A replacement with no band at all is a documented
/// coverage gap, not a violation — see this module's docs.
#[must_use]
pub fn check_frequency_hygiene(pack: &InventoryPack) -> Vec<FrequencyHygieneViolation> {
    let referenced: BTreeSet<String> = pack
        .substitution_pairs()
        .iter()
        .map(|p| p.replacement.to_lowercase())
        .collect();

    let mut violations = Vec::new();
    for band in pack.output_frequency_bands() {
        if !referenced.contains(&band.frame.to_lowercase()) {
            continue;
        }
        let reason = if !band.max.is_finite() || band.max < 0.0 {
            Some(FrequencyHygieneReason::NoFiniteNonNegativeMax)
        } else if band.sample_doc_count == 0 {
            Some(FrequencyHygieneReason::ZeroSampleDocCount)
        } else if band.max < band.observed_rate {
            Some(FrequencyHygieneReason::MaxBelowObservedRate)
        } else {
            None
        };
        if let Some(reason) = reason {
            violations.push(FrequencyHygieneViolation {
                frame: band.frame.clone(),
                reason,
            });
        }
    }
    violations
}

/// Rule 4 — guard-token exposure.
///
/// No `deletion_spans`/`substitution_pairs`/`ritual_frames`/
/// `preview_frames` pattern may have a (non-empty) literal-token content
/// that consists *entirely* of `guard_tokens` members: negation,
/// quantifiers, modals, or connectives. Those four classes are the
/// pack's own declared closed set of meaning-bearing function words: a
/// pattern reduced to nothing else is editing that class directly, which
/// this pack's other rules (disjointness, closure) have no way to
/// recognize as the sensitive edit it is.
#[must_use]
pub fn check_guard_token_exposure(pack: &InventoryPack) -> Vec<GuardTokenExposureViolation> {
    let guard: BTreeSet<String> = pack
        .guard_tokens()
        .negation
        .iter()
        .chain(pack.guard_tokens().quantifiers.iter())
        .chain(pack.guard_tokens().modals.iter())
        .chain(pack.guard_tokens().connectives.iter())
        .map(|t| t.to_lowercase())
        .collect();

    let mut candidates: Vec<(&'static str, &str, &str)> = Vec::new();
    for span in pack.deletion_spans() {
        candidates.push(("deletion_spans", &span.id, &span.pattern_source));
    }
    for sub in pack.substitution_pairs() {
        candidates.push(("substitution_pairs", &sub.id, &sub.pattern_source));
    }
    for ritual in pack.ritual_frames() {
        candidates.push(("ritual_frames", &ritual.id, &ritual.pattern_source));
    }
    for preview in pack.preview_frames() {
        candidates.push(("preview_frames", &preview.id, &preview.pattern_source));
    }
    candidates.sort_by(|a, b| a.0.cmp(b.0).then_with(|| a.1.cmp(b.1)));

    let mut violations = Vec::new();
    for (family, id, pattern_source) in candidates {
        let tokens = literal_tokens_from_pattern(pattern_source);
        if !tokens.is_empty() && tokens.iter().all(|t| guard.contains(t)) {
            violations.push(GuardTokenExposureViolation {
                family,
                id: Box::from(id),
                guard_tokens: tokens.into_iter().map(String::into_boxed_str).collect(),
            });
        }
    }
    violations
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pack_meta() -> &'static str {
        r#"
        [pack]
        version = "inventory-v1"
        corpus_manifest_sha256 = "x"
        paired_manifest_sha256 = "x"
        "#
    }

    // --- check_disjointness ---

    #[test]
    fn check_disjointness_flags_a_replacement_matching_a_deletion_span() {
        let toml = format!(
            r#"{}
            [[deletion_spans]]
            id = "span.target"
            pattern = "(?i)\\btarget\\b"
            anchor = "mid_sentence"
            repair = "delete"
            source = "seed"

            [[substitution_pairs]]
            id = "sub.only"
            pattern = "(?i)\\bfoo\\b"
            replacement = "target"
            repair = "substitute"
            source = "seed"
            "#,
            pack_meta()
        );
        let pack = InventoryPack::parse(&toml).unwrap();
        let violations = check_disjointness(&pack);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].conflicting_family, "deletion_spans");
        assert_eq!(&*violations[0].conflicting_id, "span.target");
    }

    #[test]
    fn check_disjointness_flags_a_replacement_matching_its_own_pattern() {
        let toml = format!(
            r#"{}
            [[substitution_pairs]]
            id = "sub.only"
            pattern = "(?i)\\btarget\\b"
            replacement = "target"
            repair = "substitute"
            source = "seed"
            "#,
            pack_meta()
        );
        let pack = InventoryPack::parse(&toml).unwrap();
        let violations = check_disjointness(&pack);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].conflicting_family, "substitution_pairs");
        assert_eq!(&*violations[0].conflicting_id, "sub.only");
    }

    #[test]
    fn check_disjointness_holds_for_a_clean_hermetic_pack() {
        let toml = format!(
            r#"{}
            [[deletion_spans]]
            id = "span.foo"
            pattern = "(?i)\\bfoo\\b"
            anchor = "mid_sentence"
            repair = "delete"
            source = "seed"

            [[substitution_pairs]]
            id = "sub.bar"
            pattern = "(?i)\\bbar\\b"
            replacement = "baz"
            repair = "substitute"
            source = "seed"
            "#,
            pack_meta()
        );
        let pack = InventoryPack::parse(&toml).unwrap();
        assert!(check_disjointness(&pack).is_empty());
    }

    #[test]
    fn check_disjointness_is_pairwise_and_deterministic() {
        let toml = format!(
            r#"{}
            [[deletion_spans]]
            id = "span.b"
            pattern = "(?i)\\btarget\\b"
            anchor = "mid_sentence"
            repair = "delete"
            source = "seed"

            [[deletion_spans]]
            id = "span.a"
            pattern = "(?i)\\btarget\\b"
            anchor = "mid_sentence"
            repair = "delete"
            source = "seed"

            [[substitution_pairs]]
            id = "sub.only"
            pattern = "(?i)\\btarget\\b"
            replacement = "target"
            repair = "substitute"
            source = "seed"
            "#,
            pack_meta()
        );
        let pack = InventoryPack::parse(&toml).unwrap();
        let violations = check_disjointness(&pack);
        let observed: Vec<(&str, &str)> = violations
            .iter()
            .map(|v| (v.conflicting_family, &*v.conflicting_id))
            .collect();
        assert_eq!(
            observed,
            vec![
                ("deletion_spans", "span.a"),
                ("deletion_spans", "span.b"),
                ("substitution_pairs", "sub.only"),
            ]
        );
    }

    #[test]
    fn check_disjointness_flags_a_replacement_matching_a_leading_anchored_span_only_once_embedded()
    {
        // "target please" is not, in isolation, matched by a pattern that
        // requires a trailing `\s+` after "target please" — but it would
        // be once this replacement lands back in running text, which is
        // exactly the idempotence break this check exists to catch.
        let toml = format!(
            r#"{}
            [[deletion_spans]]
            id = "span.target_please_leading"
            pattern = '(?i)^target please,?\s+'
            anchor = "sentence_initial"
            repair = "delete"
            source = "seed"

            [[substitution_pairs]]
            id = "sub.only"
            pattern = "(?i)\\bfoo\\b"
            replacement = "target please"
            repair = "substitute"
            source = "seed"
            "#,
            pack_meta()
        );
        let pack = InventoryPack::parse(&toml).unwrap();
        let violations = check_disjointness(&pack);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].conflicting_family, "deletion_spans");
        assert_eq!(&*violations[0].conflicting_id, "span.target_please_leading");
    }

    #[test]
    fn check_disjointness_flags_a_replacement_matching_a_trailing_anchored_span_only_once_embedded()
    {
        let toml = format!(
            r#"{}
            [[deletion_spans]]
            id = "span.to_suit_target_trailing"
            pattern = '(?i)\s+to suit target\b'
            anchor = "trailing"
            repair = "delete"
            source = "seed"

            [[substitution_pairs]]
            id = "sub.only"
            pattern = "(?i)\\bfoo\\b"
            replacement = "to suit target"
            repair = "substitute"
            source = "seed"
            "#,
            pack_meta()
        );
        let pack = InventoryPack::parse(&toml).unwrap();
        let violations = check_disjointness(&pack);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].conflicting_family, "deletion_spans");
        assert_eq!(
            &*violations[0].conflicting_id,
            "span.to_suit_target_trailing"
        );
    }

    #[test]
    fn check_disjointness_includes_preview_frames_in_the_candidate_set() {
        let toml = format!(
            r#"{}
            [[preview_frames]]
            id = "preview.target"
            pattern = "(?i)\\btarget\\b"
            requires_header_lemma_coverage = false
            repair = "delete"
            source = "seed"

            [[substitution_pairs]]
            id = "sub.only"
            pattern = "(?i)\\bfoo\\b"
            replacement = "target"
            repair = "substitute"
            source = "seed"
            "#,
            pack_meta()
        );
        let pack = InventoryPack::parse(&toml).unwrap();
        let violations = check_disjointness(&pack);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].conflicting_family, "preview_frames");
        assert_eq!(&*violations[0].conflicting_id, "preview.target");
    }

    // --- check_closure ---

    #[test]
    fn check_closure_flags_an_unattested_content_word() {
        let toml = format!(
            r#"{}
            [[substitution_pairs]]
            id = "sub.only"
            pattern = "(?i)\\bfoo\\b"
            replacement = "surprise"
            repair = "substitute"
            source = "seed"
            "#,
            pack_meta()
        );
        let pack = InventoryPack::parse(&toml).unwrap();
        let violations = check_closure(&pack);
        assert_eq!(violations.len(), 1);
        assert_eq!(&*violations[0].offending_tokens[0], "surprise");
    }

    #[test]
    fn check_closure_passes_for_a_literal_pattern_token() {
        let toml = format!(
            r#"{}
            [[substitution_pairs]]
            id = "sub.only"
            pattern = "(?i)\\bprior to\\b"
            replacement = "prior"
            repair = "substitute"
            source = "seed"
            "#,
            pack_meta()
        );
        let pack = InventoryPack::parse(&toml).unwrap();
        assert!(check_closure(&pack).is_empty());
    }

    #[test]
    fn check_closure_passes_via_declared_literal_tokens() {
        let toml = format!(
            r#"{}
            [[substitution_pairs]]
            id = "sub.only"
            pattern = "(?i)\\bfoo\\b"
            replacement = "surprise"
            repair = "substitute"
            source = "seed"
            declared_literal_tokens = ["surprise"]
            "#,
            pack_meta()
        );
        let pack = InventoryPack::parse(&toml).unwrap();
        assert!(check_closure(&pack).is_empty());
    }

    #[test]
    fn check_closure_passes_for_an_attested_token_with_notes() {
        let toml = format!(
            r#"{}
            [[substitution_pairs]]
            id = "sub.only"
            pattern = "(?i)\\bfoo\\b"
            replacement = "surprise"
            repair = "substitute"
            source = "seed"
            notes = "attested 9x in the human train corpus"
            attested_replacement_tokens = ["surprise"]
            "#,
            pack_meta()
        );
        let pack = InventoryPack::parse(&toml).unwrap();
        assert!(check_closure(&pack).is_empty());
    }

    #[test]
    fn check_closure_fails_for_an_attested_token_with_empty_notes() {
        let toml = format!(
            r#"{}
            [[substitution_pairs]]
            id = "sub.only"
            pattern = "(?i)\\bfoo\\b"
            replacement = "surprise"
            repair = "substitute"
            source = "seed"
            attested_replacement_tokens = ["surprise"]
            "#,
            pack_meta()
        );
        let pack = InventoryPack::parse(&toml).unwrap();
        let violations = check_closure(&pack);
        assert_eq!(violations.len(), 1);
        assert_eq!(&*violations[0].offending_tokens[0], "surprise");
    }

    #[test]
    fn check_closure_passes_for_a_function_word_allowance_entry() {
        let toml = format!(
            r#"closure_function_word_allowance = ["surprise"]

            {}
            [[substitution_pairs]]
            id = "sub.only"
            pattern = "(?i)\\bfoo\\b"
            replacement = "surprise"
            repair = "substitute"
            source = "seed"
            "#,
            pack_meta()
        );
        let pack = InventoryPack::parse(&toml).unwrap();
        assert!(check_closure(&pack).is_empty());
    }

    // --- check_frequency_hygiene ---

    fn pack_with_band(band_toml: &str) -> InventoryPack {
        let toml = format!(
            r#"{}
            [[substitution_pairs]]
            id = "sub.only"
            pattern = "(?i)\\bfoo\\b"
            replacement = "bar"
            repair = "substitute"
            source = "seed"
            declared_literal_tokens = ["bar"]

            {band_toml}
            "#,
            pack_meta()
        );
        InventoryPack::parse(&toml).unwrap()
    }

    #[test]
    fn check_frequency_hygiene_flags_nonfinite_or_negative_max() {
        let pack = pack_with_band(
            r#"[[output_frequency_bands]]
            frame = "bar"
            unit = "per_thousand_words"
            observed_rate = 0.1
            max = -1.0
            sample_doc_count = 10
            method = "test"
            "#,
        );
        let violations = check_frequency_hygiene(&pack);
        assert_eq!(violations.len(), 1);
        assert_eq!(
            violations[0].reason,
            FrequencyHygieneReason::NoFiniteNonNegativeMax
        );
    }

    #[test]
    fn check_frequency_hygiene_flags_zero_sample_doc_count() {
        let pack = pack_with_band(
            r#"[[output_frequency_bands]]
            frame = "bar"
            unit = "per_thousand_words"
            observed_rate = 0.1
            max = 0.5
            sample_doc_count = 0
            method = "test"
            "#,
        );
        let violations = check_frequency_hygiene(&pack);
        assert_eq!(violations.len(), 1);
        assert_eq!(
            violations[0].reason,
            FrequencyHygieneReason::ZeroSampleDocCount
        );
    }

    #[test]
    fn check_frequency_hygiene_flags_max_below_observed_rate() {
        let pack = pack_with_band(
            r#"[[output_frequency_bands]]
            frame = "bar"
            unit = "per_thousand_words"
            observed_rate = 0.5
            max = 0.1
            sample_doc_count = 10
            method = "test"
            "#,
        );
        let violations = check_frequency_hygiene(&pack);
        assert_eq!(violations.len(), 1);
        assert_eq!(
            violations[0].reason,
            FrequencyHygieneReason::MaxBelowObservedRate
        );
    }

    #[test]
    fn check_frequency_hygiene_ignores_a_replacement_with_no_band() {
        let toml = format!(
            r#"{}
            [[substitution_pairs]]
            id = "sub.only"
            pattern = "(?i)\\bfoo\\b"
            replacement = "bar"
            repair = "substitute"
            source = "seed"
            declared_literal_tokens = ["bar"]
            "#,
            pack_meta()
        );
        let pack = InventoryPack::parse(&toml).unwrap();
        assert!(check_frequency_hygiene(&pack).is_empty());
    }

    #[test]
    fn check_frequency_hygiene_passes_for_a_well_formed_band() {
        let pack = pack_with_band(
            r#"[[output_frequency_bands]]
            frame = "bar"
            unit = "per_thousand_words"
            observed_rate = 0.1
            max = 0.5
            sample_doc_count = 10
            method = "test"
            "#,
        );
        assert!(check_frequency_hygiene(&pack).is_empty());
    }

    // --- check_guard_token_exposure ---

    #[test]
    fn check_guard_token_exposure_flags_a_pattern_that_is_bare_negation() {
        let toml = format!(
            r#"[guard_tokens]
            negation = ["never"]

            {}
            [[substitution_pairs]]
            id = "sub.never_always"
            pattern = '(?i)\bnever\b'
            replacement = "always"
            repair = "substitute"
            source = "seed"
            declared_literal_tokens = ["always"]
            "#,
            pack_meta()
        );
        let pack = InventoryPack::parse(&toml).unwrap();
        let violations = check_guard_token_exposure(&pack);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].family, "substitution_pairs");
        assert_eq!(&*violations[0].id, "sub.never_always");
        assert_eq!(&*violations[0].guard_tokens[0], "never");
    }

    #[test]
    fn check_guard_token_exposure_ignores_a_pattern_mixing_a_guard_token_with_other_words() {
        let toml = format!(
            r#"[guard_tokens]
            modals = ["will"]

            {}
            [[deletion_spans]]
            id = "span.will_walk_you_through"
            pattern = '(?i)\bwill walk you through\b'
            anchor = "mid_sentence"
            repair = "delete"
            source = "seed"
            "#,
            pack_meta()
        );
        let pack = InventoryPack::parse(&toml).unwrap();
        assert!(check_guard_token_exposure(&pack).is_empty());
    }

    #[test]
    fn check_guard_token_exposure_holds_when_guard_tokens_section_is_absent() {
        let toml = format!(
            r#"{}
            [[deletion_spans]]
            id = "span.never"
            pattern = '(?i)\bnever\b'
            anchor = "mid_sentence"
            repair = "delete"
            source = "seed"
            "#,
            pack_meta()
        );
        let pack = InventoryPack::parse(&toml).unwrap();
        assert!(check_guard_token_exposure(&pack).is_empty());
    }

    #[test]
    fn check_guard_token_exposure_flags_a_preview_frame_that_is_bare_connective() {
        let toml = format!(
            r#"[guard_tokens]
            connectives = ["however"]

            {}
            [[preview_frames]]
            id = "preview.however"
            pattern = '(?i)\bhowever\b'
            requires_header_lemma_coverage = false
            repair = "delete"
            source = "seed"
            "#,
            pack_meta()
        );
        let pack = InventoryPack::parse(&toml).unwrap();
        let violations = check_guard_token_exposure(&pack);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].family, "preview_frames");
        assert_eq!(&*violations[0].id, "preview.however");
    }

    // --- validate ---

    #[test]
    fn validate_concatenates_rules_in_fixed_order() {
        // One violation of each kind in a single pack; `validate` must
        // return them in fixed Disjointness, Closure, FrequencyHygiene,
        // GuardTokenExposure order regardless of declaration order.
        let toml = format!(
            r#"[guard_tokens]
            connectives = ["yet"]

            {}
            [[deletion_spans]]
            id = "span.echo"
            pattern = "(?i)\\becho\\b"
            anchor = "mid_sentence"
            repair = "delete"
            source = "seed"

            [[deletion_spans]]
            id = "span.yet"
            pattern = '(?i)\byet\b'
            anchor = "mid_sentence"
            repair = "delete"
            source = "seed"

            [[substitution_pairs]]
            id = "sub.only"
            pattern = "(?i)\\bfoo\\b"
            replacement = "echo surprise"
            repair = "substitute"
            source = "seed"

            [[output_frequency_bands]]
            frame = "echo surprise"
            unit = "per_thousand_words"
            observed_rate = 0.5
            max = 0.1
            sample_doc_count = 10
            method = "test"
            "#,
            pack_meta()
        );
        let pack = InventoryPack::parse(&toml).unwrap();
        let violations = validate(&pack);
        assert_eq!(violations.len(), 4);
        assert!(matches!(violations[0], Violation::Disjointness(_)));
        assert!(matches!(violations[1], Violation::Closure(_)));
        assert!(matches!(violations[2], Violation::FrequencyHygiene(_)));
        assert!(matches!(violations[3], Violation::GuardTokenExposure(_)));
    }

    #[test]
    fn validate_on_the_embedded_inventory_v1_pack_is_clean() {
        let violations = validate(&crate::registry::INVENTORY.pack);
        assert!(
            violations.is_empty(),
            "embedded inventory-v1.toml has validation violations: {violations:#?}"
        );
    }
}
