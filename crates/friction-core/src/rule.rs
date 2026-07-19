//! [`RuleId`]: a cheap, copyable identifier for a `friction-rules` rule.

use std::fmt;

/// A cheap, copyable identifier for a rule (`friction-rules`).
///
/// Wraps a `&'static str` so rule identifiers can be embedded as constants
/// in each rule's implementation (e.g. `RuleId::new("lexical.leverage")`)
/// and passed around by value on [`crate::Patch`] and [`crate::Finding`]
/// without allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RuleId(&'static str);

impl RuleId {
    /// Creates a new rule identifier from a static string.
    #[must_use]
    pub const fn new(id: &'static str) -> Self {
        Self(id)
    }

    /// The rule identifier as a string slice.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        self.0
    }
}

impl fmt::Display for RuleId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

impl From<&'static str> for RuleId {
    fn from(id: &'static str) -> Self {
        Self::new(id)
    }
}

impl AsRef<str> for RuleId {
    fn as_ref(&self) -> &str {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `RuleId` round-trips its string and is cheaply copyable.
    #[test]
    fn rule_id_round_trips_and_copies() {
        let id = RuleId::new("lexical.leverage");
        let copy = id;
        assert_eq!(id.as_str(), "lexical.leverage");
        assert_eq!(copy.as_str(), "lexical.leverage");
        assert_eq!(id, copy);
    }

    /// `RuleId` formats via `Display` as its bare identifier string.
    #[test]
    fn rule_id_displays_as_bare_string() {
        let id = RuleId::new("connective.moreover");
        assert_eq!(id.to_string(), "connective.moreover");
    }

    /// `RuleId` orders lexicographically, so a `Vec<RuleId>` can be sorted
    /// deterministically without a `BTreeMap`.
    #[test]
    fn rule_id_orders_lexicographically() {
        let mut ids = vec![RuleId::new("b"), RuleId::new("a"), RuleId::new("c")];
        ids.sort();
        assert_eq!(
            ids,
            vec![RuleId::new("a"), RuleId::new("b"), RuleId::new("c")]
        );
    }

    /// `RuleId` converts from `&'static str` via `From`/`Into`.
    #[test]
    fn rule_id_from_static_str() {
        let id: RuleId = "structural.unbullet".into();
        assert_eq!(id.as_str(), "structural.unbullet");
    }
}
