//! Shared model metadata and lexical-shortlist support for Marian backends.

mod manifest;
mod shortlist;

pub use manifest::{Architecture, Checksums, ModelManifest};
pub use shortlist::LexicalShortlist;
