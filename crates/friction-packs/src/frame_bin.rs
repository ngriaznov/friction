//! Derived binary frame-pack artifact: the compiled rule program
//! serialized flat, read back as a zero-copy view.
//!
//! Same cure as `dms_bin` and the perceptron weights: run the
//! expensive step — TOML parsing, the rejection gauntlet, attestation
//! rate derivation — **once, offline**, in `corpus-tool frame-pack`,
//! and hand the runtime a layout it can use straight off the
//! `include_bytes!`'d static with a handful of slice splits. The frame
//! compile itself is not slow, but its attestation input is the 2.5 MB
//! DMS index TOML, and re-parsing that at every process start would
//! hand back the startup cost the DMS binary pack just removed.
//!
//! # Layout
//!
//! A fixed header, then little-endian sections in a fixed order. Every
//! multi-byte field is read via `from_le_bytes` on a byte slice —
//! alignment-free by construction, so no `unsafe` is ever needed.
//!
//! - **Header**: 8-byte magic `FRFRMPCK`, `u16` format version, 64-byte
//!   lowercase-hex sha256 of the source rules TOML. The recorded sha
//!   lets a stale `.bin` fail loudly in the drift test rather than
//!   silently serving rules compiled from an older set.
//! - **Interner**: word count `n`; `n + 1` cumulative `u32` byte
//!   offsets into a UTF-8 pool; `u32` pool length; the pool. Words are
//!   sorted, so an id **is** its sorted rank and lookup is a binary
//!   search over the offsets. The compiler assigns first-use ids; the
//!   serializer sorts and remaps every reference once, here.
//! - **Class names**: same string-table shape (`u16` count).
//! - **Classes**: per class, `u16` member count, then per member `u8`
//!   id count and that many `u32` interner ids (multi-word members
//!   match as phrases). A `u32` cumulative byte-offset table in front
//!   gives O(1) access per class.
//! - **Pattern ops / template ops**: one shared 6-byte cell shape —
//!   `code: u8`, `a: u8`, `b: u32` — in two blobs. Pattern op codes
//!   are [`PatOp`]'s discriminants with the high bit of `code` marking
//!   an optional element; template op codes are [`TplOp`]'s.
//! - **Anchor ids**: one flat `u32` array; rules reference slices.
//! - **Rule ids**: one UTF-8 pool; rules reference byte ranges.
//! - **Rules**: fixed [`RULE_RECORD_LEN`]-byte records (offsets/lengths
//!   into the blobs above, kind, flags, support, measured rates).
//!
//! # Determinism
//!
//! [`pack_frame_bin`] is a pure function of the compiled pack: word
//! remapping is by sorted order, every section is written in
//! declaration order, and no hash-based collection is iterated. Packing
//! the same compile twice produces bit-identical bytes (pinned by this
//! module's own test).

use crate::frame_compile::{CPat, CTpl, CompiledKind, CompiledPack};
use crate::frame_rules::{Clitic, Slot, Tag};

/// The 8-byte magic every frame pack starts with.
const MAGIC: [u8; 8] = *b"FRFRMPCK";

/// On-disk format version: bumped if the layout changes shape.
const FORMAT_VERSION: u16 = 1;

/// Length of the lowercase-hex sha256 recorded in the header.
const SHA256_HEX_LEN: usize = 64;

/// The fixed-length prefix: magic + version + source sha256 hex.
const HEADER_LEN: usize = 8 + 2 + SHA256_HEX_LEN;

/// Bytes per serialized op cell: `code: u8` + `a: u8` + `b: u32`.
const OP_CELL_LEN: usize = 6;

/// The high bit of an op cell's `code`, marking an optional pattern
/// element.
const OPTIONAL_BIT: u8 = 0x80;

/// Bytes per serialized rule record.
const RULE_RECORD_LEN: usize = 42;

/// One decoded pattern op. Group structure is delimiter-encoded
/// ([`Self::GroupStart`]/[`Self::AltSep`]/[`Self::GroupEnd`]) so the
/// blob stays flat. The matcher walks it with a cursor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatOp {
    /// Lemma-matched literal (interner id).
    Lit(u32),
    /// Bare tag position.
    Tag(Tag),
    /// Tag position restricted to a class's members.
    Class {
        /// Required tag.
        tag: Tag,
        /// Class index.
        class: u16,
    },
    /// Content slot.
    Slot(Slot),
    /// Match must start its sentence.
    SentStart,
    /// Match must end its sentence.
    SentEnd,
    /// Contraction clitic.
    Clitic(Clitic),
    /// Opens an alternation group of `alts` alternatives.
    GroupStart {
        /// Alternative count.
        alts: u8,
    },
    /// Separates two alternatives inside a group.
    AltSep,
    /// Closes the innermost group.
    GroupEnd,
}

/// One decoded template op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TplOp {
    /// Emit this interned literal.
    Lit(u32),
    /// Copy a slot's captured bytes.
    Slot(Slot),
    /// Copy the token matched at the pattern element with this tag.
    TagCopy(Tag),
    /// Realize the lemma in the form of the token matched at pattern
    /// element `agree_elem` (fallback: the bare lemma).
    Inflect {
        /// Interner id of the lemma.
        lemma: u32,
        /// Pattern element index supplying the form.
        agree_elem: u8,
    },
    /// Realize the lemma in this tag's own form.
    InflectForm {
        /// Interner id of the lemma.
        lemma: u32,
        /// The form to realize.
        tag: Tag,
    },
    /// Realize the matched class member in this tag's form.
    ClassRealize {
        /// Class index.
        class: u16,
        /// The form to realize.
        tag: Tag,
    },
    /// Copy the token matched by the pattern's group with this ordinal.
    AltCopy(u8),
    /// Emit this clitic.
    Clitic(Clitic),
}

/// Errors reading a frame pack artifact. Never expected for the
/// vendored artifact (covered by round-trip and drift tests): only
/// for corrupted or hand-edited input.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FrameBinError {
    /// The artifact is shorter than a section requires.
    #[error("frame pack truncated at {section}")]
    Truncated {
        /// Which section ran out of bytes.
        section: &'static str,
    },
    /// The artifact does not start with the frame-pack magic.
    #[error("bad frame pack magic")]
    BadMagic,
    /// The artifact's format version is not this module's.
    #[error("unsupported frame pack version {0}")]
    UnsupportedVersion(u16),
    /// A section holds bytes that are not valid UTF-8.
    #[error("frame pack {section} holds invalid UTF-8")]
    InvalidUtf8 {
        /// Which section held the bytes.
        section: &'static str,
    },
    /// An op cell's code byte is not a known op.
    #[error("unknown op code {code} in {section}")]
    UnknownOp {
        /// The unknown code byte.
        code: u8,
        /// Which blob held it.
        section: &'static str,
    },
}

/// Serializes a compiled pack. `source_sha256_hex` is the lowercase
/// hex sha256 of the rules TOML the pack was compiled from.
///
/// Word ids are remapped here from the compiler's first-use order to
/// sorted order, so the on-disk interner supports binary search.
#[must_use]
pub fn pack_frame_bin(pack: &CompiledPack, source_sha256_hex: &str) -> Vec<u8> {
    assert_eq!(
        source_sha256_hex.len(),
        SHA256_HEX_LEN,
        "source sha must be lowercase hex sha256"
    );
    // Sorted remap: new id = rank of the word in sorted order.
    let mut sorted: Vec<(String, u32)> = pack.interner.iter().cloned().zip(0u32..).collect();
    sorted.sort();
    let mut remap = vec![0u32; pack.interner.len()];
    for (new_id, (_, old_id)) in sorted.iter().enumerate() {
        remap[*old_id as usize] = u32::try_from(new_id).expect("word count fits u32");
    }
    let map = |id: u32| remap[id as usize];

    let mut out = Vec::new();
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(source_sha256_hex.as_bytes());

    // Interner.
    write_string_table(&mut out, sorted.iter().map(|(word, _)| word.as_str()));
    // Class names. Kept in the on-disk format for format stability and
    // external tooling even though `FramePackView` has no reader for it —
    // dropping the table would bump FORMAT_VERSION and force regenerating
    // every shipped artifact.
    write_string_table(&mut out, pack.class_names.iter().map(String::as_str));
    // Classes: per-class byte offsets, then the member blob.
    let mut class_blob = Vec::new();
    let mut class_offsets = vec![0u32];
    for members in &pack.classes {
        class_blob.extend_from_slice(
            &u16::try_from(members.len())
                .expect("member count")
                .to_le_bytes(),
        );
        for member in members {
            class_blob.push(u8::try_from(member.len()).expect("member word count"));
            for id in member {
                class_blob.extend_from_slice(&map(*id).to_le_bytes());
            }
        }
        class_offsets.push(u32::try_from(class_blob.len()).expect("class blob length"));
    }
    out.extend_from_slice(
        &u16::try_from(pack.classes.len())
            .expect("class count")
            .to_le_bytes(),
    );
    for offset in &class_offsets {
        out.extend_from_slice(&offset.to_le_bytes());
    }
    out.extend_from_slice(&u32::try_from(class_blob.len()).expect("len").to_le_bytes());
    out.extend_from_slice(&class_blob);

    let sections = write_rules(pack, map);
    for blob in [
        &sections.pat_ops,
        &sections.tpl_ops,
        &sections.anchor_ids,
        &sections.id_pool,
    ] {
        out.extend_from_slice(
            &u32::try_from(blob.len())
                .expect("blob length")
                .to_le_bytes(),
        );
        out.extend_from_slice(blob);
    }
    out.extend_from_slice(
        &u32::try_from(pack.rules.len())
            .expect("rule count")
            .to_le_bytes(),
    );
    out.extend_from_slice(&sections.records);
    out
}

/// The per-rule blobs of the pack, written in one pass.
struct RuleSections {
    pat_ops: Vec<u8>,
    tpl_ops: Vec<u8>,
    anchor_ids: Vec<u8>,
    id_pool: Vec<u8>,
    records: Vec<u8>,
}

/// Writes every rule's ops, anchor, id, and fixed record.
fn write_rules(pack: &CompiledPack, map: impl Fn(u32) -> u32 + Copy) -> RuleSections {
    let mut pat_ops: Vec<u8> = Vec::new();
    let mut tpl_ops: Vec<u8> = Vec::new();
    let mut anchor_ids: Vec<u8> = Vec::new();
    let mut id_pool: Vec<u8> = Vec::new();
    let mut records: Vec<u8> = Vec::new();
    for rule in &pack.rules {
        let pat_off = u32::try_from(pat_ops.len() / OP_CELL_LEN).expect("pattern op offset");
        let mut pat_len = 0u16;
        for elem in &rule.pattern {
            pat_len += write_pat_elem(&mut pat_ops, elem, map, false);
        }
        let tpl_off = u32::try_from(tpl_ops.len() / OP_CELL_LEN).expect("template op offset");
        for op in &rule.template {
            write_tpl_op(&mut tpl_ops, op, map);
        }
        let tpl_len = u16::try_from(rule.template.len()).expect("template length");
        let anchor_off = u32::try_from(anchor_ids.len() / 4).expect("anchor offset");
        for id in &rule.anchor_ids {
            anchor_ids.extend_from_slice(&map(*id).to_le_bytes());
        }
        let id_off = u32::try_from(id_pool.len()).expect("id pool offset");
        id_pool.extend_from_slice(rule.id.as_bytes());

        records.extend_from_slice(&id_off.to_le_bytes());
        records.extend_from_slice(
            &u16::try_from(rule.id.len())
                .expect("rule id length")
                .to_le_bytes(),
        );
        records.extend_from_slice(&pat_off.to_le_bytes());
        records.extend_from_slice(&pat_len.to_le_bytes());
        records.extend_from_slice(&tpl_off.to_le_bytes());
        records.extend_from_slice(&tpl_len.to_le_bytes());
        records.extend_from_slice(&anchor_off.to_le_bytes());
        records.extend_from_slice(
            &u16::try_from(rule.anchor_ids.len())
                .expect("anchor length")
                .to_le_bytes(),
        );
        records.extend_from_slice(&rule.anchor_elems.0.to_le_bytes());
        records.extend_from_slice(&rule.anchor_elems.1.to_le_bytes());
        records.push(kind_code(rule.kind));
        records.push(u8::from(rule.surface_match));
        records.extend_from_slice(&rule.support.to_le_bytes());
        #[expect(clippy::cast_possible_truncation, reason = "rates are display data")]
        {
            records.extend_from_slice(&(rule.m_pm as f32).to_le_bytes());
            records.extend_from_slice(&(rule.h_pm as f32).to_le_bytes());
        }
    }
    RuleSections {
        pat_ops,
        tpl_ops,
        anchor_ids,
        id_pool,
        records,
    }
}

/// Writes one string table: `u32` count, `count + 1` cumulative `u32`
/// offsets, `u32` pool length, pool bytes.
fn write_string_table<'a>(out: &mut Vec<u8>, strings: impl Iterator<Item = &'a str>) {
    let strings: Vec<&str> = strings.collect();
    out.extend_from_slice(
        &u32::try_from(strings.len())
            .expect("string count")
            .to_le_bytes(),
    );
    let mut offset = 0u32;
    out.extend_from_slice(&offset.to_le_bytes());
    for s in &strings {
        offset += u32::try_from(s.len()).expect("string length");
        out.extend_from_slice(&offset.to_le_bytes());
    }
    out.extend_from_slice(&offset.to_le_bytes());
    for s in &strings {
        out.extend_from_slice(s.as_bytes());
    }
}

/// Writes one pattern element as op cells; returns how many cells.
fn write_pat_elem(
    out: &mut Vec<u8>,
    elem: &CPat,
    map: impl Fn(u32) -> u32 + Copy,
    optional: bool,
) -> u16 {
    let opt_bit = if optional { OPTIONAL_BIT } else { 0 };
    let mut cell = |code: u8, a: u8, b: u32| {
        out.push(code | opt_bit);
        out.push(a);
        out.extend_from_slice(&b.to_le_bytes());
    };
    match elem {
        CPat::Lit(id) => cell(0, 0, map(*id)),
        CPat::Tag(tag) => cell(1, tag.code(), 0),
        CPat::Class { tag, class } => cell(2, tag.code(), u32::from(*class)),
        CPat::Slot(slot) => cell(3, slot.code(), 0),
        CPat::SentStart => cell(4, 0, 0),
        CPat::SentEnd => cell(5, 0, 0),
        CPat::Clitic(clitic) => cell(6, clitic.code(), 0),
        CPat::Opt(inner) => return write_pat_elem(out, inner, map, true),
        CPat::Group { alts, optional } => {
            let opt_bit = if *optional { OPTIONAL_BIT } else { 0 };
            out.push(7 | opt_bit);
            out.push(u8::try_from(alts.len()).expect("alternative count"));
            out.extend_from_slice(&0u32.to_le_bytes());
            let mut cells: u16 = 1;
            for (i, alt) in alts.iter().enumerate() {
                if i > 0 {
                    out.push(8);
                    out.push(0);
                    out.extend_from_slice(&0u32.to_le_bytes());
                    cells += 1;
                }
                for inner in alt {
                    cells += write_pat_elem(out, inner, map, false);
                }
            }
            out.push(9);
            out.push(0);
            out.extend_from_slice(&0u32.to_le_bytes());
            return cells + 1;
        }
    }
    1
}

/// Writes one template op cell.
fn write_tpl_op(out: &mut Vec<u8>, op: &CTpl, map: impl Fn(u32) -> u32) {
    let mut cell = |code: u8, a: u8, b: u32| {
        out.push(code);
        out.push(a);
        out.extend_from_slice(&b.to_le_bytes());
    };
    match op {
        CTpl::Lit(id) => cell(0, 0, map(*id)),
        CTpl::Slot(slot) => cell(1, slot.code(), 0),
        CTpl::TagCopy(tag) => cell(2, tag.code(), 0),
        CTpl::Inflect { lemma, agree_elem } => cell(3, *agree_elem, map(*lemma)),
        CTpl::InflectForm { lemma, tag } => cell(4, tag.code(), map(*lemma)),
        CTpl::ClassRealize { class, tag } => cell(5, tag.code(), u32::from(*class)),
        CTpl::AltCopy(ordinal) => cell(6, *ordinal, 0),
        CTpl::Clitic(clitic) => cell(7, clitic.code(), 0),
    }
}

/// The kind byte written into rule records.
const fn kind_code(kind: CompiledKind) -> u8 {
    match kind {
        CompiledKind::Rewrite => 0,
        CompiledKind::Delete => 1,
        CompiledKind::Guard => 2,
        CompiledKind::Report => 3,
    }
}

/// A zero-copy view over a serialized frame pack.
#[derive(Debug, Clone, Copy)]
pub struct FramePackView<'a> {
    source_sha256_hex: &'a str,
    word_offsets: &'a [u8],
    word_pool: &'a str,
    word_count: u32,
    class_offsets: &'a [u8],
    class_blob: &'a [u8],
    class_count: u16,
    pat_ops: &'a [u8],
    tpl_ops: &'a [u8],
    anchor_ids: &'a [u8],
    id_pool: &'a str,
    records: &'a [u8],
    rule_count: u32,
}

/// One rule as viewed from the pack.
#[derive(Debug, Clone)]
pub struct RuleView<'a> {
    /// The rule id.
    pub id: &'a str,
    /// The rule kind.
    pub kind: CompiledKind,
    /// Whether literals match lowercased surfaces instead of lemmas.
    pub surface_match: bool,
    /// Decoded pattern ops.
    pub pattern: PatOps<'a>,
    /// Decoded template ops.
    pub template: TplOps<'a>,
    /// The anchor's word-id sequence.
    pub anchor: AnchorIds<'a>,
    /// Element index range `(start, len)` of the anchor within the
    /// pattern.
    pub anchor_elems: (u16, u16),
    /// Conflict tie-break weight.
    pub support: u32,
    /// Measured machine per-million rate.
    pub m_pm: f32,
    /// Measured human per-million rate.
    pub h_pm: f32,
}

/// Iterator over a rule's pattern op cells; each item is
/// `(op, optional)`.
#[derive(Debug, Clone)]
pub struct PatOps<'a> {
    cells: &'a [u8],
}

impl Iterator for PatOps<'_> {
    type Item = Result<(PatOp, bool), FrameBinError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cells.is_empty() {
            return None;
        }
        let (cell, rest) = self.cells.split_at(OP_CELL_LEN);
        self.cells = rest;
        Some(decode_pat_cell(cell))
    }
}

impl PatOps<'_> {
    /// Number of op cells remaining.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.cells.len() / OP_CELL_LEN
    }

    /// Whether no op cells remain.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }
}

/// Iterator over a rule's template op cells.
#[derive(Debug, Clone)]
pub struct TplOps<'a> {
    cells: &'a [u8],
}

impl Iterator for TplOps<'_> {
    type Item = Result<TplOp, FrameBinError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cells.is_empty() {
            return None;
        }
        let (cell, rest) = self.cells.split_at(OP_CELL_LEN);
        self.cells = rest;
        Some(decode_tpl_cell(cell))
    }
}

/// A rule's anchor id sequence, decoded on the fly.
#[derive(Debug, Clone)]
pub struct AnchorIds<'a> {
    bytes: &'a [u8],
}

impl Iterator for AnchorIds<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.bytes.is_empty() {
            return None;
        }
        let (id, rest) = self.bytes.split_at(4);
        self.bytes = rest;
        Some(u32::from_le_bytes(id.try_into().expect("4-byte split")))
    }
}

impl AnchorIds<'_> {
    /// Number of ids remaining.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.bytes.len() / 4
    }

    /// Whether no ids remain.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

fn decode_pat_cell(cell: &[u8]) -> Result<(PatOp, bool), FrameBinError> {
    let optional = cell[0] & OPTIONAL_BIT != 0;
    let code = cell[0] & !OPTIONAL_BIT;
    let a = cell[1];
    let b = u32::from_le_bytes(cell[2..6].try_into().expect("cell is 6 bytes"));
    let unknown = || FrameBinError::UnknownOp {
        code,
        section: "pattern ops",
    };
    let op = match code {
        0 => PatOp::Lit(b),
        1 => PatOp::Tag(Tag::from_code(a).ok_or_else(unknown)?),
        2 => PatOp::Class {
            tag: Tag::from_code(a).ok_or_else(unknown)?,
            class: u16::try_from(b).map_err(|_| unknown())?,
        },
        3 => PatOp::Slot(Slot::from_code(a).ok_or_else(unknown)?),
        4 => PatOp::SentStart,
        5 => PatOp::SentEnd,
        6 => PatOp::Clitic(Clitic::from_code(a).ok_or_else(unknown)?),
        7 => PatOp::GroupStart { alts: a },
        8 => PatOp::AltSep,
        9 => PatOp::GroupEnd,
        _ => return Err(unknown()),
    };
    Ok((op, optional))
}

fn decode_tpl_cell(cell: &[u8]) -> Result<TplOp, FrameBinError> {
    let code = cell[0];
    let a = cell[1];
    let b = u32::from_le_bytes(cell[2..6].try_into().expect("cell is 6 bytes"));
    let unknown = || FrameBinError::UnknownOp {
        code,
        section: "template ops",
    };
    let op = match code {
        0 => TplOp::Lit(b),
        1 => TplOp::Slot(Slot::from_code(a).ok_or_else(unknown)?),
        2 => TplOp::TagCopy(Tag::from_code(a).ok_or_else(unknown)?),
        3 => TplOp::Inflect {
            lemma: b,
            agree_elem: a,
        },
        4 => TplOp::InflectForm {
            lemma: b,
            tag: Tag::from_code(a).ok_or_else(unknown)?,
        },
        5 => TplOp::ClassRealize {
            class: u16::try_from(b).map_err(|_| unknown())?,
            tag: Tag::from_code(a).ok_or_else(unknown)?,
        },
        6 => TplOp::AltCopy(a),
        7 => TplOp::Clitic(Clitic::from_code(a).ok_or_else(unknown)?),
        _ => return Err(unknown()),
    };
    Ok(op)
}

/// A cursor over the artifact's bytes, erroring (never panicking) on
/// truncation.
struct Reader<'a> {
    bytes: &'a [u8],
}

impl<'a> Reader<'a> {
    const fn take(&mut self, len: usize, section: &'static str) -> Result<&'a [u8], FrameBinError> {
        if self.bytes.len() < len {
            return Err(FrameBinError::Truncated { section });
        }
        let (taken, rest) = self.bytes.split_at(len);
        self.bytes = rest;
        Ok(taken)
    }

    fn u16(&mut self, section: &'static str) -> Result<u16, FrameBinError> {
        Ok(u16::from_le_bytes(
            self.take(2, section)?.try_into().expect("2-byte split"),
        ))
    }

    fn u32(&mut self, section: &'static str) -> Result<u32, FrameBinError> {
        Ok(u32::from_le_bytes(
            self.take(4, section)?.try_into().expect("4-byte split"),
        ))
    }

    fn str(&mut self, len: usize, section: &'static str) -> Result<&'a str, FrameBinError> {
        std::str::from_utf8(self.take(len, section)?)
            .map_err(|_| FrameBinError::InvalidUtf8 { section })
    }
}

impl<'a> FramePackView<'a> {
    /// Parses the artifact's sections into a view, slice splits only.
    ///
    /// # Errors
    /// Returns [`FrameBinError`] on truncation, bad magic, an
    /// unsupported version, or invalid UTF-8 in a string section.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, FrameBinError> {
        if bytes.len() < HEADER_LEN {
            return Err(FrameBinError::Truncated { section: "header" });
        }
        let mut r = Reader { bytes };
        let magic = r.take(8, "header")?;
        if magic != MAGIC {
            return Err(FrameBinError::BadMagic);
        }
        let version = r.u16("header")?;
        if version != FORMAT_VERSION {
            return Err(FrameBinError::UnsupportedVersion(version));
        }
        let source_sha256_hex = r.str(SHA256_HEX_LEN, "header")?;

        let (word_count, word_offsets, word_pool) = read_string_table(&mut r, "interner")?;
        // Class names are format-stable for tooling but have no runtime
        // reader here; parse and drop the table just to advance past it.
        read_string_table(&mut r, "class names")?;

        let class_count = r.u16("classes")?;
        let class_offsets = r.take((usize::from(class_count) + 1) * 4, "classes")?;
        let class_blob_len = r.u32("classes")?;
        let class_blob = r.take(class_blob_len as usize, "classes")?;

        let pat_len = r.u32("pattern ops")?;
        let pat_ops = r.take(pat_len as usize, "pattern ops")?;
        let tpl_len = r.u32("template ops")?;
        let tpl_ops = r.take(tpl_len as usize, "template ops")?;
        let anchor_len = r.u32("anchor ids")?;
        let anchor_ids = r.take(anchor_len as usize, "anchor ids")?;
        let id_pool_len = r.u32("rule ids")?;
        let id_pool = r.str(id_pool_len as usize, "rule ids")?;
        let rule_count = r.u32("rules")?;
        let records = r.take(rule_count as usize * RULE_RECORD_LEN, "rules")?;

        Ok(Self {
            source_sha256_hex,
            word_offsets,
            word_pool,
            word_count,
            class_offsets,
            class_blob,
            class_count,
            pat_ops,
            tpl_ops,
            anchor_ids,
            id_pool,
            records,
            rule_count,
        })
    }

    /// The lowercase-hex sha256 of the rules TOML this pack was
    /// compiled from.
    #[must_use]
    pub const fn source_sha256_hex(&self) -> &'a str {
        self.source_sha256_hex
    }

    /// Number of interned words.
    #[must_use]
    pub const fn word_count(&self) -> u32 {
        self.word_count
    }

    /// The word with this id, if in range.
    #[must_use]
    pub fn word(&self, id: u32) -> Option<&'a str> {
        string_at(self.word_offsets, self.word_pool, self.word_count, id)
    }

    /// The id of this word, by binary search (words are sorted).
    ///
    /// Only used by this module's own tests — the crate's one production
    /// consumer (`friction_match::frame_rewrite::FrameIndex`) builds its
    /// own hash-map lookup instead, so this stays private.
    #[cfg(test)]
    #[must_use]
    fn word_id(&self, word: &str) -> Option<u32> {
        let mut lo = 0u32;
        let mut hi = self.word_count;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            match self.word(mid)?.cmp(word) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid),
            }
        }
        None
    }

    /// Number of classes.
    #[must_use]
    pub const fn class_count(&self) -> u16 {
        self.class_count
    }

    /// The members of this class: each member is a word-id sequence.
    #[must_use]
    pub fn class_members(&self, class: u16) -> Option<Vec<Vec<u32>>> {
        if class >= self.class_count {
            return None;
        }
        let at = usize::from(class) * 4;
        let start = u32::from_le_bytes(self.class_offsets[at..at + 4].try_into().ok()?) as usize;
        let mut r = Reader {
            bytes: self.class_blob.get(start..)?,
        };
        let member_count = r.u16("classes").ok()?;
        let mut members = Vec::with_capacity(usize::from(member_count));
        for _ in 0..member_count {
            let word_count = *r.take(1, "classes").ok()?.first()?;
            let mut member = Vec::with_capacity(usize::from(word_count));
            for _ in 0..word_count {
                member.push(r.u32("classes").ok()?);
            }
            members.push(member);
        }
        Some(members)
    }

    /// Number of rules.
    #[must_use]
    pub const fn rule_count(&self) -> u32 {
        self.rule_count
    }

    /// The rule with this index.
    #[must_use]
    pub fn rule(&self, index: u32) -> Option<RuleView<'a>> {
        if index >= self.rule_count {
            return None;
        }
        let at = index as usize * RULE_RECORD_LEN;
        let rec = &self.records[at..at + RULE_RECORD_LEN];
        let u16_at = |o: usize| u16::from_le_bytes(rec[o..o + 2].try_into().expect("in record"));
        let u32_at = |o: usize| u32::from_le_bytes(rec[o..o + 4].try_into().expect("in record"));
        let id_off = u32_at(0) as usize;
        let id_len = usize::from(u16_at(4));
        let pat_off = u32_at(6) as usize;
        let pat_len = usize::from(u16_at(10));
        let tpl_off = u32_at(12) as usize;
        let tpl_len = usize::from(u16_at(16));
        let anchor_off = u32_at(18) as usize;
        let anchor_len = usize::from(u16_at(22));
        let anchor_elems = (u16_at(24), u16_at(26));
        let kind = match rec[28] {
            0 => CompiledKind::Rewrite,
            1 => CompiledKind::Delete,
            2 => CompiledKind::Guard,
            _ => CompiledKind::Report,
        };
        let surface_match = rec[29] != 0;
        let support = u32_at(30);
        let m_pm = f32::from_le_bytes(rec[34..38].try_into().expect("in record"));
        let h_pm = f32::from_le_bytes(rec[38..42].try_into().expect("in record"));
        Some(RuleView {
            id: self.id_pool.get(id_off..id_off + id_len)?,
            kind,
            surface_match,
            pattern: PatOps {
                cells: self
                    .pat_ops
                    .get(pat_off * OP_CELL_LEN..(pat_off + pat_len) * OP_CELL_LEN)?,
            },
            template: TplOps {
                cells: self
                    .tpl_ops
                    .get(tpl_off * OP_CELL_LEN..(tpl_off + tpl_len) * OP_CELL_LEN)?,
            },
            anchor: AnchorIds {
                bytes: self
                    .anchor_ids
                    .get(anchor_off * 4..(anchor_off + anchor_len) * 4)?,
            },
            anchor_elems,
            support,
            m_pm,
            h_pm,
        })
    }
}

/// Reads one string table written by [`write_string_table`].
fn read_string_table<'a>(
    r: &mut Reader<'a>,
    section: &'static str,
) -> Result<(u32, &'a [u8], &'a str), FrameBinError> {
    let count = r.u32(section)?;
    let offsets = r.take((count as usize + 1) * 4, section)?;
    let pool_len = r.u32(section)?;
    let pool = r.str(pool_len as usize, section)?;
    Ok((count, offsets, pool))
}

/// The string with this id from an offsets-and-pool table.
fn string_at<'a>(offsets: &[u8], pool: &'a str, count: u32, id: u32) -> Option<&'a str> {
    if id >= count {
        return None;
    }
    let at = id as usize * 4;
    let start = u32::from_le_bytes(offsets[at..at + 4].try_into().ok()?) as usize;
    let end = u32::from_le_bytes(offsets[at + 4..at + 8].try_into().ok()?) as usize;
    pool.get(start..end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_compile::{CorpusEvidence, compile};
    use crate::frame_rules::FrameRuleSet;

    fn shipped_pack() -> CompiledPack {
        let set = FrameRuleSet::parse(include_str!("../packs/frame-rules-en-v1.toml"))
            .expect("shipped rules parse");
        let rates = CorpusEvidence::from_dms_toml(include_str!("../packs/dms-index-en-v1.toml"))
            .expect("shipped dms index parses");
        compile(&set, &rates).expect("shipped set compiles").0
    }

    const TEST_SHA: &str = "0000000000000000000000000000000000000000000000000000000000000000";

    /// Round trip: every rule's id, kind, ops, anchor, and metadata
    /// survive serialization, with word ids remapped consistently.
    #[test]
    fn round_trip_preserves_every_rule() {
        let pack = shipped_pack();
        let bytes = pack_frame_bin(&pack, TEST_SHA);
        let view = FramePackView::parse(&bytes).expect("pack parses");
        assert_eq!(view.rule_count() as usize, pack.rules.len());
        assert_eq!(view.word_count() as usize, pack.interner.len());
        assert_eq!(usize::from(view.class_count()), pack.classes.len());
        for (i, rule) in pack.rules.iter().enumerate() {
            let v = view.rule(u32::try_from(i).expect("index")).expect("rule");
            assert_eq!(v.id, rule.id);
            assert_eq!(v.kind, rule.kind);
            assert_eq!(v.surface_match, rule.surface_match);
            assert_eq!(v.support, rule.support);
            assert_eq!(v.anchor_elems, rule.anchor_elems);
            // Anchor words match through the remap.
            let anchor_words: Vec<&str> = v
                .anchor
                .map(|id| view.word(id).expect("anchor word"))
                .collect();
            let source_words: Vec<&str> = rule
                .anchor_ids
                .iter()
                .map(|id| pack.interner[*id as usize].as_str())
                .collect();
            assert_eq!(anchor_words, source_words, "rule {}", rule.id);
            // Ops decode without error and match in count (groups add
            // delimiter cells, so pattern count is >=).
            let pat_ops = v
                .pattern
                .clone()
                .inspect(|cell| {
                    cell.as_ref().expect("pattern op decodes");
                })
                .count();
            assert!(pat_ops >= rule.pattern.len(), "rule {}", rule.id);
            let tpl_ops = v
                .template
                .clone()
                .inspect(|op| {
                    op.as_ref().expect("template op decodes");
                })
                .count();
            assert_eq!(tpl_ops, rule.template.len(), "rule {}", rule.id);
        }
    }

    /// Word lookup is a working binary search over the sorted table.
    #[test]
    fn word_id_binary_search_agrees_with_word() {
        let pack = shipped_pack();
        let bytes = pack_frame_bin(&pack, TEST_SHA);
        let view = FramePackView::parse(&bytes).expect("pack parses");
        for id in 0..view.word_count() {
            let word = view.word(id).expect("word");
            assert_eq!(view.word_id(word), Some(id), "word {word:?}");
        }
        assert_eq!(view.word_id("zzz-not-a-word"), None);
    }

    /// Packing the same compile twice is bit-identical.
    #[test]
    fn packing_is_deterministic() {
        let pack = shipped_pack();
        assert_eq!(
            pack_frame_bin(&pack, TEST_SHA),
            pack_frame_bin(&pack, TEST_SHA)
        );
    }

    /// Truncation and corruption fail loudly, never panic.
    #[test]
    fn corrupted_input_errors_instead_of_panicking() {
        let pack = shipped_pack();
        let bytes = pack_frame_bin(&pack, TEST_SHA);
        assert!(matches!(
            FramePackView::parse(&bytes[..HEADER_LEN - 1]),
            Err(FrameBinError::Truncated { .. })
        ));
        let mut bad_magic = bytes.clone();
        bad_magic[0] = b'X';
        assert!(matches!(
            FramePackView::parse(&bad_magic),
            Err(FrameBinError::BadMagic)
        ));
        let mut bad_version = bytes;
        bad_version[8] = 0xFF;
        assert!(matches!(
            FramePackView::parse(&bad_version),
            Err(FrameBinError::UnsupportedVersion(_))
        ));
    }

    /// Class members survive the trip, phrases intact.
    #[test]
    fn class_members_round_trip() {
        let pack = shipped_pack();
        let bytes = pack_frame_bin(&pack, TEST_SHA);
        let view = FramePackView::parse(&bytes).expect("pack parses");
        for (i, members) in pack.classes.iter().enumerate() {
            let class = u16::try_from(i).expect("class index");
            let viewed = view.class_members(class).expect("class");
            let viewed_words: Vec<Vec<&str>> = viewed
                .iter()
                .map(|m| {
                    m.iter()
                        .map(|id| view.word(*id).expect("member word"))
                        .collect()
                })
                .collect();
            let source_words: Vec<Vec<&str>> = members
                .iter()
                .map(|m| {
                    m.iter()
                        .map(|id| pack.interner[*id as usize].as_str())
                        .collect()
                })
                .collect();
            assert_eq!(viewed_words, source_words, "class {i}");
        }
    }
}
