Nice use of `nom` for the INI-style config format — the combinator structure reads cleanly and mirrors the grammar well: `section_header` wrapping `delimited(char('['), take_until("]"), char(']'))`, then `many0(key_value)` for the body. A couple of small things worth fixing.

`take_until("]")` inside the section header will happily accept an empty section name (`[]`), and it'll also silently accept a name containing characters you probably don't want there, like leading/trailing whitespace which then ends up baked into the section key. I'd swap that for `take_while1` over an explicit allowed-character set so empty and malformed names fail parsing instead of producing a section named `""`.

Your `key_value` parser trims the value with `.trim()` after parsing but doesn't trim the key, so `foo = bar` produces a key of `"foo "` rather than `"foo"` — easy to miss since it'll work fine until someone writes a config file with a space before the `=`, which people do constantly by habit. Trim both sides.

Also worth double-checking: `alt((comment, key_value, section_header))` tries `comment` first, but your comment parser matches from `;` to end-of-line and doesn't consume the trailing newline, so the next `many0` iteration sees a leading `\n` and (depending on how your whitespace parser is written) may or may not treat that as a separator. Worth a unit test with a comment line immediately followed by a key-value line to pin this down either way.

None of this is a big deal — the overall design is solid and idiomatic `nom`, and error handling via `nom::Err` propagating up through `?` is exactly what I'd expect. Fix the trimming and the empty-section-name gap and I'd merge this.
