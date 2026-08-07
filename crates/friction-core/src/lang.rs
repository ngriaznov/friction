//! The language a document's prose is analyzed as.
//!
//! `En` is the only member in v1: every pack, weight artifact, and word
//! table currently shipped is English (the `-en` tag on the data files).
//! The enum exists so that pack selection, engine loading, and corpus
//! partitioning all key on one type from the start — adding a language
//! means adding a variant here and letting the compiler surface every
//! seam that needs a decision, instead of hunting hardcoded `"en"`
//! strings.

use std::fmt;
use std::str::FromStr;

/// A supported prose language, keyed by its BCP-47 primary tag.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Lang {
    /// English — the only shipped language in v1.
    #[default]
    En,
}

impl Lang {
    /// The BCP-47 primary language tag (`"en"`), matching the `-en` tag
    /// on pack data files and the `lang` field of corpus manifest
    /// records.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::En => "en",
        }
    }
}

impl fmt::Display for Lang {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Error for [`Lang::from_str`]: the tag names no supported language.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnknownLang(pub String);

impl fmt::Display for UnknownLang {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unsupported language tag {:?} (supported: en)", self.0)
    }
}

impl std::error::Error for UnknownLang {}

impl FromStr for Lang {
    type Err = UnknownLang;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "en" => Ok(Self::En),
            other => Err(UnknownLang(other.to_owned())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_its_tag() {
        assert_eq!("en".parse::<Lang>().unwrap(), Lang::En);
        assert_eq!(Lang::En.as_str(), "en");
        assert_eq!(Lang::En.to_string(), "en");
    }

    #[test]
    fn rejects_an_unknown_tag() {
        let err = "tlh".parse::<Lang>().unwrap_err();
        assert_eq!(err.0, "tlh");
    }

    #[test]
    fn defaults_to_english() {
        assert_eq!(Lang::default(), Lang::En);
    }
}
