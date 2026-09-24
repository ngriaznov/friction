//! End-to-end coverage for construction-level frame rules and the two
//! frame-pass policies they depend on: a one-word guard does not veto a
//! construction that names the guarded word, and a gate-held candidate
//! never shadows the smaller edits it overlaps.

use friction_edit::Engine;

fn engine() -> Engine {
    Engine::new().expect("embedded packs and models load")
}

fn fix(source: &str) -> String {
    engine().fix_document(source).expect("engine runs").0
}

/// The additive correlative collapses to a plain conjunction, even
/// though "also" carries a one-word guard (`conng.also`).
#[test]
fn not_only_but_also_becomes_a_conjunction() {
    assert_eq!(
        fix("This method not only saves time but also ensures accuracy in channel programming.\n"),
        "This method saves time and ensures accuracy in channel programming.\n"
    );
}

/// The comma before "but also" goes with it.
#[test]
fn not_only_comma_but_also_drops_the_comma() {
    assert_eq!(
        fix(
            "Grants let us not only better meet the current needs of the project, but also \
             help push it into the future.\n"
        ),
        "Grants let us better meet the current needs of the project and help push it into \
         the future.\n"
    );
}

/// "merely" is the same additive correlative.
#[test]
fn not_merely_but_also_becomes_a_conjunction() {
    assert_eq!(
        fix("The tool is not merely a formatter but also a linter.\n"),
        "The tool is a formatter and a linter.\n"
    );
}

/// Without the literal "but also" the frame is left alone: a bare
/// "not only A, but B" can be corrective, and the inverted "not only
/// does X, it also Y" has no second-limb anchor.
#[test]
fn not_only_without_but_also_is_untouched() {
    for source in [
        "In the summer program we improve not only the language, but the whole ecosystem.\n",
        "Not only does the cache reduce latency, it also cuts database load.\n",
        "It is not only fast.\n",
    ] {
        assert_eq!(fix(source), source);
    }
}

/// A seam-held construction must not keep the smaller rewrite inside
/// its slot from firing: here the long rule holds, and "enhances" ->
/// "improves" still applies.
#[test]
fn held_construction_does_not_shadow_inner_rewrite() {
    let source = "This strategic configuration not only enhances performance but also ensures \
                  that applications remain responsive under load.\n";
    let fixed = fix(source);
    assert!(
        fixed.contains("not only improves performance"),
        "the inner rewrite must apply: {fixed}"
    );
}

/// `simply` is a deletable intensifier, but `simply put` is an idiom:
/// deleting its first word strands "put".
#[test]
fn simply_put_is_not_split() {
    for source in [
        "Simply put, the cache is a map with a clock.\n",
        "The cache, simply put, is a map with a clock.\n",
    ] {
        assert_eq!(fix(source), source);
    }
    assert_eq!(
        fix("You can simply run the script.\n"),
        "You can run the script.\n"
    );
}

/// "a handful of" is the same quantity as "a few", in a plainer word.
#[test]
fn a_handful_of_becomes_a_few() {
    assert_eq!(
        fix("We fetch data from a handful of internal services.\n"),
        "We fetch data from a few internal services.\n"
    );
}

/// A negated sentence's closing "at all" only stamps the negation.
#[test]
fn negated_at_all_is_deleted() {
    assert_eq!(
        fix("Yet the Scan method does not expose an error at all.\n"),
        "Yet the Scan method does not expose an error.\n"
    );
    assert_eq!(
        fix("That traffic never touches the router at all.\n"),
        "That traffic never touches the router.\n"
    );
}

/// Outside a negation "at all" carries meaning, "if at all" is an idiom,
/// and a mid-sentence "at all" is left for the reader.
#[test]
fn at_all_outside_a_negated_close_is_untouched() {
    for source in [
        "That is the trick that makes the method practical at all.\n",
        "It is not clear, if at all.\n",
        "The cache is not used at all, and writes go straight through.\n",
    ] {
        assert_eq!(fix(source), source);
    }
}
