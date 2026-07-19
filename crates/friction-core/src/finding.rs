//! [`Finding`]: a detected issue, diagnostic or backing a [`crate::Patch`].

use std::ops::Range;

use crate::error::CoreError;
use crate::patch::Tier;
use crate::rule::RuleId;
use crate::span::{self, Spanned};

/// A detected issue at a span in the source, produced by a rule's `scan`
/// step (`friction-rules`) ahead of — or instead of — an automatically
/// applicable [`crate::Patch`].
///
/// Every `Finding` is rendered as a `miette` diagnostic with a labeled
/// span; `Fix`-tier findings additionally carry a corresponding `Patch`,
/// while `Suggest`-tier findings are diagnostic-only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    /// The rule that produced this finding.
    pub rule: RuleId,
    /// Byte range in the original source this finding is about.
    pub range: Range<usize>,
    /// Human-readable description, surfaced via `miette` diagnostics.
    pub message: String,
    /// Meaning-preservation tier.
    pub tier: Tier,
}

impl Finding {
    /// Creates a new finding.
    #[must_use]
    pub fn new(rule: RuleId, range: Range<usize>, message: impl Into<String>, tier: Tier) -> Self {
        Self {
            rule,
            range,
            message: message.into(),
            tier,
        }
    }

    /// Validates this finding's `range` against `source`.
    ///
    /// # Errors
    /// Returns [`CoreError`] if `range` is out of bounds for `source` or
    /// splits a UTF-8 character.
    pub fn validate(&self, source: &str) -> Result<(), CoreError> {
        span::validate_range(source, &self.range)
    }
}

impl Spanned for Finding {
    fn range(&self) -> Range<usize> {
        self.range.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A finding reports its own range via `Spanned` and validates against
    /// a source that contains it.
    #[test]
    fn finding_validate_accepts_in_bounds_range() {
        let finding = Finding::new(
            RuleId::new("symmetry.triad"),
            0..5,
            "triad coordination pattern",
            Tier::Suggest,
        );
        assert!(finding.validate("hello world").is_ok());
        assert_eq!(finding.range(), 0..5);
        assert_eq!(finding.message, "triad coordination pattern");
        assert_eq!(finding.tier, Tier::Suggest);
    }

    /// An out-of-bounds finding range is rejected.
    #[test]
    fn finding_validate_rejects_out_of_bounds_range() {
        let finding = Finding::new(RuleId::new("symmetry.triad"), 0..50, "oops", Tier::Fix);
        assert!(finding.validate("hello").is_err());
    }
}
