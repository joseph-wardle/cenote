//! pbrt-v4 scene importer: a `.pbrt` file in, a cenote [`ChangeSet`] out.
//!
//! This crate is a *client* of cenote's public scene API — it consumes
//! nothing internal, so its existence proves the change-set surface is
//! sufficient for a real importer (the same forcing function the future
//! wire protocol needs). The pipeline is three layers, one module each:
//! `tokenize` (pbrt's four token shapes, zero-copy), `parse` (directives
//! with typed parameter lists, `Include` spliced), and `map` (pbrt's
//! graphics-state semantics lowered onto the change-set schema).
//!
//! Fidelity over coverage: the supported subset is the one the published
//! corpus actually uses, and everything outside it **warns by token
//! name** — silence never means "handled". The five semantic traps that
//! make naive importers subtly wrong — photometric light normalization,
//! `roughness`→α remapping, shorter-axis `fov`, left-handed coordinates
//! (with `ReverseOrientation`), and equal-area octahedral sky images —
//! are each handled and each pinned by a test (`map`'s module doc walks
//! through all five).
//!
//! [`ChangeSet`]: cenote::scene::changeset::ChangeSet

use std::path::Path;

use cenote::Result;
use cenote::scene::changeset::ChangeSet;

mod env;
mod map;
mod parse;
mod tokenize;

/// An imported scene: the change-set, and every fidelity warning the
/// import raised — parameters dropped, features degraded, tokens
/// skipped. Callers surface these (the CLI prints them, the viewer
/// logs them); an import that warns still renders.
pub struct Import {
    /// The scene as an apply-ready change-set: every referenced path is
    /// absolute, so it applies directly — serialize through
    /// [`cenote::format`] to write it as a `.ron` scene file.
    pub set: ChangeSet,
    /// Human-readable fidelity warnings, in encounter order, `file:line`
    /// prefixed where one location is to blame.
    pub warnings: Vec<String>,
}

/// Import a pbrt-v4 scene file.
///
/// `generated` names a writable directory for assets the import must
/// *derive* (a resampled octahedral sky, a constant-sky EXR) — source
/// images and PLY files are referenced where they are, never copied.
///
/// # Errors
///
/// [`cenote::Error::SceneFormat`] with `file:line` for anything
/// malformed — unreadable files, unknown directives, type-mismatched
/// parameters — and [`cenote::Error::Scene`] when a derived asset can't
/// be built. Unsupported-but-well-formed content is a warning, not an
/// error.
pub fn import(scene: &Path, generated: &Path) -> Result<Import> {
    std::fs::create_dir_all(generated)?;
    let parser = parse::Parser::open(scene)?;
    let stem = scene.file_stem().map_or_else(
        || "scene".to_owned(),
        |stem| stem.to_string_lossy().into_owned(),
    );
    let (set, warnings) = map::Mapper::new(parser, generated, stem).run()?;
    Ok(Import { set, warnings })
}
