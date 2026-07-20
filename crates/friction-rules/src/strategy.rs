//! [`StrategyRng`]: the small, deterministic pseudo-random generator a
//! [`crate::Rule`]'s `fix` step uses to choose among several
//! meaning-preserving fix strategies.
//!
//! Seeded from a hash of the sentence being fixed and the rule's own id,
//! never from wall-clock time or a shared global generator: the same
//! sentence, fixed by the same rule, always makes the same strategy choice
//! on any machine and any run (determinism); a different sentence, or a
//! different rule, makes an independent choice (so the tool never develops
//! a single constant tic — always deleting a filler phrase the same way,
//! say — which would just be a different, self-inflicted statistical
//! fingerprint).

use friction_core::RuleId;
use xxhash_rust::xxh64::xxh64;

/// A splitmix64-based pseudo-random generator, seeded deterministically
/// from a sentence's bytes and a rule id.
///
/// This is a strategy-*selection* helper, not a cryptographic or
/// statistical-quality generator — it exists only so a rule with more than
/// one meaning-preserving fix strategy for the same finding (e.g. "delete
/// the connective and recapitalize" vs. "swap it for a shorter one") can
/// pick between them without either breaking determinism (ambient
/// randomness) or always picking the same one everywhere (a fixed
/// constant, which just trades one detectable pattern for another).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrategyRng {
    state: u64,
}

impl StrategyRng {
    /// Seeds a generator from `sentence_bytes` (the exact source bytes of
    /// the sentence being fixed) and `rule` (the fixing rule's id).
    ///
    /// The two are hashed together with `xxh64`: `sentence_bytes`, then a
    /// single `0x00` separator byte, then the rule id's own UTF-8 bytes —
    /// the separator keeps two different `(sentence, rule)` pairs whose
    /// naive concatenation would otherwise collide (e.g. sentence `"ab"`
    /// with rule id `"c"` vs. sentence `"a"` with rule id `"bc"`) from
    /// producing the same seed. The resulting hash seeds a splitmix64
    /// generator directly.
    #[must_use]
    pub fn seeded(sentence_bytes: &[u8], rule: RuleId) -> Self {
        Self::from_seed(seed_hash(sentence_bytes, rule))
    }

    /// Builds a generator directly from a 64-bit seed, bypassing
    /// `xxh64`. Exposed for tests that want to check the underlying
    /// splitmix64 sequence itself against an independently computed
    /// reference value.
    #[must_use]
    pub const fn from_seed(seed: u64) -> Self {
        Self { state: seed }
    }

    /// The next pseudo-random 64-bit value, advancing internal state.
    ///
    /// Implements splitmix64 (Steele, Lea & Flood 2014 / the generator
    /// used by, among others, Java's `SplittableRandom` and Rust's `rand`
    /// crate as a seed-mixing step): a golden-ratio increment followed by
    /// two xor-shift/multiply rounds, chosen for being small, dependency-
    /// free, and fast to hand-verify against published reference vectors
    /// (see this module's tests) rather than for cryptographic strength,
    /// which strategy selection does not need.
    pub const fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A pseudo-random index in `0..bound`, or `0` if `bound` is `0`.
    ///
    /// Uses plain modulo reduction rather than a bias-corrected scheme
    /// (e.g. Lemire's method): every use in this workspace picks among a
    /// handful of fixed fix strategies (single digits), where the
    /// resulting modulo bias is far below anything a document-level style
    /// metric could ever pick up, and a rejection-sampling loop would add
    /// review surface without a measurable benefit.
    pub const fn gen_range(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        #[allow(clippy::cast_possible_truncation)]
        let index = (self.next_u64() % bound as u64) as usize;
        index
    }

    /// Picks one element of `options` pseudo-randomly, or `None` if
    /// `options` is empty.
    pub fn choose<'a, T>(&mut self, options: &'a [T]) -> Option<&'a T> {
        if options.is_empty() {
            None
        } else {
            options.get(self.gen_range(options.len()))
        }
    }
}

/// Hashes `sentence_bytes` and `rule` together into one 64-bit seed. See
/// [`StrategyRng::seeded`] for the exact byte layout.
fn seed_hash(sentence_bytes: &[u8], rule: RuleId) -> u64 {
    let rule_bytes = rule.as_str().as_bytes();
    let mut buf = Vec::with_capacity(sentence_bytes.len() + 1 + rule_bytes.len());
    buf.extend_from_slice(sentence_bytes);
    buf.push(0);
    buf.extend_from_slice(rule_bytes);
    xxh64(&buf, 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(id: &'static str) -> RuleId {
        RuleId::new(id)
    }

    /// The splitmix64 sequence for seed `0`, independently computed from
    /// the reference algorithm (Steele, Lea & Flood 2014) in a standalone
    /// Python implementation — not derived by running this module's own
    /// code — so this test can catch a transcription bug in
    /// [`StrategyRng::next_u64`] rather than merely confirming it agrees
    /// with itself.
    #[test]
    fn next_u64_matches_independently_computed_splitmix64_reference_seed_0() {
        let mut rng = StrategyRng::from_seed(0);
        let expected: [u64; 5] = [
            16_294_208_416_658_607_535,
            7_960_286_522_194_355_700,
            487_617_019_471_545_679,
            17_909_611_376_780_542_444,
            1_961_750_202_426_094_747,
        ];
        for value in expected {
            assert_eq!(rng.next_u64(), value);
        }
    }

    /// Same reference check for seed `42`, to rule out a formula that
    /// happens to work only for the seed-zero special case (e.g. an
    /// accidentally-elided `wrapping_add`).
    #[test]
    fn next_u64_matches_independently_computed_splitmix64_reference_seed_42() {
        let mut rng = StrategyRng::from_seed(42);
        let expected: [u64; 3] = [
            13_679_457_532_755_275_413,
            2_949_826_092_126_892_291,
            5_139_283_748_462_763_858,
        ];
        for value in expected {
            assert_eq!(rng.next_u64(), value);
        }
    }

    /// Same `(sentence, rule)` pair always seeds the identical sequence —
    /// the determinism half of the invariant.
    #[test]
    fn seeded_same_inputs_produce_the_same_sequence() {
        let mut a = StrategyRng::seeded(b"The kit includes screws.", rule("lexical.leverage"));
        let mut b = StrategyRng::seeded(b"The kit includes screws.", rule("lexical.leverage"));
        for _ in 0..8 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    /// Different sentences (same rule) seed different sequences — checked
    /// as pairwise distinctness of the first output across several
    /// sentences, a property of the fixed `xxh64` hash function under
    /// test, not a statistical/flaky assertion.
    #[test]
    fn seeded_different_sentences_produce_different_first_output() {
        let sentences: [&[u8]; 8] = [
            b"The kit includes screws.",
            b"The kit includes bolts.",
            b"Moreover, it just works.",
            b"However, it still fails.",
            b"We shipped it on Tuesday.",
            b"They shipped it on Monday.",
            b"It leverages a robust pipeline.",
            b"It uses a simple pipeline.",
        ];
        let mut firsts: Vec<u64> = sentences
            .iter()
            .map(|s| StrategyRng::seeded(s, rule("lexical.leverage")).next_u64())
            .collect();
        let before = firsts.len();
        firsts.sort_unstable();
        firsts.dedup();
        assert_eq!(firsts.len(), before, "expected all first outputs distinct");
    }

    /// Different rules (same sentence) seed different sequences.
    #[test]
    fn seeded_different_rules_produce_different_first_output() {
        let sentence = b"It leverages a robust pipeline.";
        let a = StrategyRng::seeded(sentence, rule("lexical.leverage")).next_u64();
        let b = StrategyRng::seeded(sentence, rule("connective.moreover")).next_u64();
        assert_ne!(a, b);
    }

    /// The `(sentence_bytes, rule_id)` seed avoids the naive-concatenation
    /// collision: `("ab", "c")` and `("a", "bc")` must not seed the same
    /// sequence.
    #[test]
    fn seeded_avoids_concatenation_collision() {
        let a = StrategyRng::seeded(b"ab", rule("c")).next_u64();
        let b = StrategyRng::seeded(b"a", rule("bc")).next_u64();
        assert_ne!(a, b);
    }

    /// `gen_range(0)` returns `0` rather than dividing by zero.
    #[test]
    fn gen_range_zero_bound_returns_zero() {
        let mut rng = StrategyRng::from_seed(7);
        assert_eq!(rng.gen_range(0), 0);
    }

    /// `gen_range` always stays within `0..bound`.
    #[test]
    fn gen_range_stays_within_bound() {
        let mut rng = StrategyRng::seeded(b"some sentence.", rule("rhythm.split"));
        for _ in 0..200 {
            let value = rng.gen_range(5);
            assert!(value < 5, "{value} not in 0..5");
        }
    }

    /// `choose` returns `None` for an empty slice and an in-bounds
    /// reference for a non-empty one.
    #[test]
    fn choose_handles_empty_and_non_empty_slices() {
        let mut rng = StrategyRng::from_seed(99);
        let empty: [&str; 0] = [];
        assert_eq!(rng.choose(&empty), None);

        let options = ["delete", "swap", "none"];
        for _ in 0..20 {
            let chosen = rng.choose(&options).expect("non-empty slice");
            assert!(options.contains(chosen));
        }
    }
}
