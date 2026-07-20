//! Concrete [`crate::Rule`] implementations, one module per
//! [`crate::RuleFamily`].
//!
//! Each family module is self-contained: its own matching tables, its own
//! span/strategy logic, its own tests. This module only wires them
//! together as submodules of `friction-rules`.

pub mod connective;
pub mod contraction;
pub mod lexical;
pub mod rhythm;
pub mod structural;
pub mod symmetry;
