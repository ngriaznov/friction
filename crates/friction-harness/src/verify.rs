//! Reusable per-fixture assertion logic: [`check_accept_pair`] for `accept`
//! fixtures, [`check_reject_output`] for `reject` fixtures.

use crate::closure::{ClosureVerdict, check_closure};
use crate::error::HarnessError;
use crate::score::{Scorer, Verdict};
use crate::tellspan::count_tell_spans;

/// The result of checking one `accept`-fixture (input, output) pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AcceptCheck {
    /// Whether every token the output introduces traces back to the input.
    pub closure: ClosureVerdict,
    /// The input's tell-span count.
    pub input_tell_count: usize,
    /// The output's tell-span count.
    pub output_tell_count: usize,
    /// The output's [`Score`](crate::score::Score) compared against the
    /// input's.
    pub verdict: Verdict,
}

/// Checks one `accept`-fixture (input, output) pair: closure, tell-span
/// counts on both sides, and the output's score against the input's.
///
/// # Errors
/// Propagates [`Scorer::score`]'s errors.
pub fn check_accept_pair(
    scorer: &Scorer,
    input: &str,
    output: &str,
) -> Result<AcceptCheck, HarnessError> {
    let closure = check_closure(input, output, scorer.tagger());
    let input_tell_count = count_tell_spans(input, scorer.tagger());
    let output_tell_count = count_tell_spans(output, scorer.tagger());

    let input_score = scorer.score(input)?;
    let output_score = scorer.score(output)?;
    let verdict = output_score.compare_to(&input_score);

    Ok(AcceptCheck {
        closure,
        input_tell_count,
        output_tell_count,
        verdict,
    })
}

/// Which mechanism, if any, flags a `reject`-fixture's documented bad
/// output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejectFlag {
    /// Closure caught it: the bad output introduces a token the input
    /// (plus its per-input licensed vocabulary) does not account for.
    ClosureViolated(Vec<String>),
    /// Closure held, but the bad output's score is not better than the
    /// input's.
    ScoreNotBetter(Verdict),
    /// Neither mechanism flagged it — a real test failure the calling test
    /// must fail on; kept as data (not a panic) so the test controls the
    /// failure message and can include the fixture id.
    Unflagged,
}

/// Checks one `reject`-fixture's documented bad output against `input`:
/// closure first, then (only if closure holds) score comparison.
///
/// # Errors
/// Propagates [`Scorer::score`]'s errors.
pub fn check_reject_output(
    scorer: &Scorer,
    input: &str,
    bad_output: &str,
) -> Result<RejectFlag, HarnessError> {
    match check_closure(input, bad_output, scorer.tagger()) {
        ClosureVerdict::Violated { excess_tokens } => {
            Ok(RejectFlag::ClosureViolated(excess_tokens))
        }
        ClosureVerdict::Holds => {
            let input_score = scorer.score(input)?;
            let bad_output_score = scorer.score(bad_output)?;
            let verdict = bad_output_score.compare_to(&input_score);
            if verdict == Verdict::Better {
                Ok(RejectFlag::Unflagged)
            } else {
                Ok(RejectFlag::ScoreNotBetter(verdict))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::shared_scorer;

    #[test]
    fn check_accept_pair_reports_closure_counts_and_verdict() {
        let scorer = shared_scorer();
        let input = "This guide will walk you through the setup.";
        let output = "This guide covers the setup.";
        let check = check_accept_pair(scorer, input, output).expect("scoring succeeds");
        assert_eq!(check.closure, ClosureVerdict::Holds);
        assert!(check.output_tell_count < check.input_tell_count);
        assert_eq!(check.verdict, Verdict::Better);
    }

    #[test]
    fn check_reject_output_flags_closure_violation() {
        let scorer = shared_scorer();
        let input = "Initially, the biggest hurdle was simply getting used to it.";
        let bad_output = "Initially, the biggest hurdle was to getting used to it.";
        let flag = check_reject_output(scorer, input, bad_output).expect("scoring succeeds");
        assert!(matches!(flag, RejectFlag::ClosureViolated(_)));
    }
}
