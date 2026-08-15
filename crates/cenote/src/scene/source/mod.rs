//! Bulk-payload readers and prep caches: the file formats a description
//! can reference — PLY meshes, `.hair` grooms, `NanoVDB` grids, texture
//! images — each resolved at scene prep by `super::lower`.

pub(crate) mod hair;
pub(crate) mod ply;
pub(crate) mod texture;
pub(crate) mod vdb;
