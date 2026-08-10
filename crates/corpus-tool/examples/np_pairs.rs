//! Experiment scaffolding (not shipped): emit NP-internal compound
//! candidates — adjacent `JJ|NN|NNS` → `NN|NNS` word pairs, lowercased —
//! one per line, from every file passed on argv. Feeds the general
//! jargon-detection feasibility measurement; consumed by shell tooling
//! that checks the pairs against the attested-bigram evidence set.

use friction_core::TokenKind;
use friction_nlp::{PerceptronTagger, Tagger};
use std::fs;

fn main() {
    let tagger = PerceptronTagger::new().expect("embedded tagger");
    for path in std::env::args().skip(1) {
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        for sentence in text.split_inclusive(['.', '!', '?', '\n', ';', ':']) {
            let sentence_tags = tagger.tag(sentence, 0);
            let words: Vec<(&str, &str)> = sentence_tags
                .iter()
                .filter(|t| t.token.kind == TokenKind::Word)
                .map(|t| (&sentence[t.token.range.clone()], t.pos.as_str()))
                .collect();
            for pair in words.windows(2) {
                let [(w1, p1), (w2, p2)] = pair else { continue };
                let modifier = matches!(*p1, "JJ" | "NN" | "NNS");
                let head = matches!(*p2, "NN" | "NNS");
                // Proper nouns (NNP/NNPS) excluded on both sides: names
                // are unattested by nature, never invented terminology.
                if modifier && head && w1.len() >= 3 && w2.len() >= 3 {
                    println!("{} {}", w1.to_lowercase(), w2.to_lowercase());
                }
            }
        }
    }
}
