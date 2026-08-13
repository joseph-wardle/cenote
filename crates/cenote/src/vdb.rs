//! `NanoVDB` grid residency: the `.nvdb` file parser, the `.vdb` prep cache,
//! and the GPU grid pool.
//!
//! The renderer core reads only `.nvdb` — `NanoVDB`'s GPU-native serialization,
//! whose uncompressed grid payload is exactly the bytes `PNanoVDB` addresses
//! on the GPU. A `.vdb` reference goes through [`prepared`], which shells
//! out to the optional `cenote-vdb-prep` tool (vdb-prep/ in the checkout;
//! `OpenVDB` stays behind that process boundary) and caches the result
//! beside the source, exactly as texture prep caches its DDS files:
//! content-hashed, so touched mtimes stay hits and edits re-prep.
//!
//! [`GridPool`] is where grids live on the GPU: one grow-only
//! device-local buffer the kernels see as a `StructuredBuffer<uint>`,
//! each grid at a 32-byte-aligned byte offset that doubles as its handle.
//! `PNanoVDB` addresses are 32-bit byte offsets, which caps the pool at
//! 4 GiB.

use std::collections::HashMap;
use std::fs;
use std::io::{Read as _, Seek as _, SeekFrom};
use std::path::{Path, PathBuf};

use ash::vk;

use crate::error::{Error, Result};
use crate::gpu::{Buffer, Context, MemoryLocation};

/// The `.nvdb` file magic (`"NanoVDB0"`), shared by files written before the
/// split into per-purpose magics.
const MAGIC_NUMB: u64 = 0x3042_4456_6f6e_614e;
/// The in-memory grid magic (`"NanoVDB1"`) — what a raw grid buffer may
/// open with under the new numbering.
const MAGIC_GRID: u64 = 0x3142_4456_6f6e_614e;
/// The file-header magic (`"NanoVDB2"`) under the new numbering.
const MAGIC_FILE: u64 = 0x3242_4456_6f6e_614e;

/// `nanovdb::GridType::Float` — the one type the pipeline handles today.
/// The rest of the scalar family (`Half`, `Fp4`/`8`/`16`, `FpN`) joins when a rung
/// needs it; the names here are for error messages.
pub const GRID_TYPE_FLOAT: u32 = 1;

/// `nanovdb::GridType` names, indexed by value — error-message material.
const GRID_TYPE_NAMES: &[&str] = &[
    "Unknown", "Float", "Double", "Int16", "Int32", "Int64", "Vec3f", "Vec3d", "Mask", "Half",
    "UInt32", "Boolean", "RGBA8", "Fp4", "Fp8", "Fp16", "FpN", "Vec4f", "Vec4d", "Index",
    "OnIndex", "IndexMask", "OnIndexMask", "PointIndex", "Vec3u8", "Vec3u16", "UInt8",
];

/// `NanoVDB`'s required in-memory grid alignment; pool offsets honor it.
const DATA_ALIGNMENT: u64 = 32;

/// One grid's metadata as the `.nvdb` container records it
/// (`nanovdb::io::FileMetaData` plus the name that follows it), and where
/// its raw payload sits in the file.
#[derive(Clone, Debug)]
pub struct GridMeta {
    /// The grid's name as authored (`OpenVDB` grid name, e.g. "density").
    pub name: String,
    /// `nanovdb::GridType` as stored; see [`GRID_TYPE_FLOAT`].
    pub grid_type: u32,
    /// The in-memory (and, uncompressed, on-disk) grid size in bytes.
    pub grid_size: u64,
    /// Index-space AABB of active voxels, min ijk then max ijk (inclusive).
    pub index_bbox: [i32; 6],
    /// Byte offset of the raw grid payload within the file.
    pub file_offset: u64,
}

impl GridMeta {
    /// The grid type's `NanoVDB` name, for error messages.
    #[must_use]
    pub fn type_name(&self) -> &'static str {
        GRID_TYPE_NAMES
            .get(self.grid_type as usize)
            .copied()
            .unwrap_or("out-of-range")
    }
}

fn scene_error(path: &Path, message: impl std::fmt::Display) -> Error {
    Error::Scene(format!("volume \"{}\": {message}", path.display()))
}

/// Read `N` bytes from `file` or fail with a truncation error.
fn read_exact<const N: usize>(file: &mut fs::File, path: &Path) -> Result<[u8; N]> {
    let mut bytes = [0u8; N];
    file.read_exact(&mut bytes)
        .map_err(|e| scene_error(path, format!("truncated .nvdb: {e}")))?;
    Ok(bytes)
}

/// Parse every grid's metadata from an uncompressed `.nvdb` without
/// touching the payloads. The container is a sequence of segments, each a
/// 16-byte header, the grids' 176-byte metadata records with their names,
/// then the raw grid payloads in the same order.
///
/// # Errors
///
/// [`Error::Scene`] on a bad magic, a compressed codec (the prep tool
/// writes uncompressed only — the GPU maps these bytes as-is), or a
/// truncated file.
pub fn read_metadata(path: &Path) -> Result<Vec<GridMeta>> {
    let mut file =
        fs::File::open(path).map_err(|e| scene_error(path, format!("can't open: {e}")))?;
    let file_len = file
        .metadata()
        .map_err(|e| scene_error(path, format!("can't stat: {e}")))?
        .len();
    let mut grids = Vec::new();
    let mut cursor = 0u64;
    while cursor < file_len {
        // Segment header: magic u64, version u32, gridCount u16, codec u16.
        let header = read_exact::<16>(&mut file, path)?;
        let magic = u64::from_le_bytes(header[0..8].try_into().unwrap());
        if magic != MAGIC_NUMB && magic != MAGIC_FILE {
            return Err(scene_error(
                path,
                format!("not a .nvdb file (magic {magic:#018x})"),
            ));
        }
        let grid_count = u16::from_le_bytes(header[12..14].try_into().unwrap());
        let codec = u16::from_le_bytes(header[14..16].try_into().unwrap());
        if codec != 0 {
            return Err(scene_error(
                path,
                "compressed .nvdb (ZIP/Blosc) — re-prep with cenote-vdb-prep, \
                 which writes the uncompressed layout the GPU maps directly",
            ));
        }
        cursor += 16;
        let mut segment = Vec::with_capacity(usize::from(grid_count));
        for _ in 0..grid_count {
            // nanovdb::io::FileMetaData, 176 bytes, little-endian, then the
            // NUL-terminated grid name of the recorded length.
            let meta = read_exact::<176>(&mut file, path)?;
            let word = |at: usize| u64::from_le_bytes(meta[at..at + 8].try_into().unwrap());
            let grid_size = word(0);
            let file_size = word(8);
            let grid_type = u32::from_le_bytes(meta[32..36].try_into().unwrap());
            let mut index_bbox = [0i32; 6];
            for (i, slot) in index_bbox.iter_mut().enumerate() {
                *slot = i32::from_le_bytes(meta[88 + 4 * i..92 + 4 * i].try_into().unwrap());
            }
            let name_size = u32::from_le_bytes(meta[136..140].try_into().unwrap());
            if file_size != grid_size {
                return Err(scene_error(
                    path,
                    "grid payload is compressed — cenote-vdb-prep writes uncompressed only",
                ));
            }
            let mut name = vec![0u8; name_size as usize];
            file.read_exact(&mut name)
                .map_err(|e| scene_error(path, format!("truncated grid name: {e}")))?;
            cursor += 176 + u64::from(name_size);
            let name = String::from_utf8_lossy(&name)
                .trim_end_matches('\0')
                .to_owned();
            segment.push(GridMeta {
                name,
                grid_type,
                grid_size,
                index_bbox,
                file_offset: 0, // patched below, once the names are past
            });
        }
        // Payloads follow the last name, in metadata order.
        for meta in &mut segment {
            meta.file_offset = cursor;
            cursor += meta.grid_size;
        }
        if cursor > file_len {
            return Err(scene_error(
                path,
                format!("truncated .nvdb: payloads reach {cursor} of {file_len} bytes"),
            ));
        }
        file.seek(SeekFrom::Start(cursor))
            .map_err(|e| scene_error(path, format!("seek failed: {e}")))?;
        grids.append(&mut segment);
    }
    Ok(grids)
}

// `GridData`/`TreeData`/`RootData` and node byte offsets for ABI 32.7,
// mirrored from the vendored `PNanoVDB.h` (`PNANOVDB_GRID_SIZE`,
// `PNANOVDB_GRID_OFF_MAP`, `PNANOVDB_MAP_OFF_*`, `PNANOVDB_TREE_OFF_*`,
// `PNANOVDB_*_OFF_CHILD_MASK`, and the float row of
// `pnanovdb_grid_type_constants`). The header is the authority; these drift
// only if the vendored copy does, and `float_layout_matches_pnanovdb` pins
// the arithmetic that depends on them.
const GRID_DATA_SIZE: u64 = 672;
const GRID_OFF_MAP: usize = 296;
const MAP_OFF_MATF: usize = 0;
const MAP_OFF_INVMATF: usize = 36;
const MAP_OFF_VECF: usize = 72;
const GRID_OFF_GRID_TYPE: usize = 636;
const TREE_DATA_SIZE: u64 = 64;
const TREE_OFF_NODE_OFFSET_LEAF: usize = 0;
const TREE_OFF_NODE_OFFSET_LOWER: usize = 8;
const TREE_OFF_NODE_OFFSET_UPPER: usize = 16;
const TREE_OFF_NODE_OFFSET_ROOT: usize = 24;
const TREE_OFF_NODE_COUNT_LEAF: usize = 32;
const TREE_OFF_NODE_COUNT_LOWER: usize = 36;
const TREE_OFF_NODE_COUNT_UPPER: usize = 40;
/// Float-grid `RootData`: the tile count, then the background value; the
/// tiles follow the 64-byte header, 32 bytes each.
const ROOT_OFF_TABLE_SIZE: usize = 24;
const ROOT_OFF_BACKGROUND: usize = 28;
const ROOT_SIZE: u64 = 64;
const ROOT_TILE_SIZE: u64 = 32;
const ROOT_TILE_OFF_CHILD: usize = 8;
const ROOT_TILE_OFF_VALUE: usize = 20;

/// One grid's in-payload header, parsed on the CPU: the metadata the
/// container records, the grid's own index↔world map as affine rows, the
/// background value, and where the tree's node arrays sit. Everything prep
/// needs to derive a heterogeneous medium's shell, transform, and majorant
/// lattice.
///
/// "Asset space" here is the grid's own world space — the space its `.vdb`
/// was authored in, which an instance transform then places in the scene.
#[derive(Clone, Debug)]
pub struct GridHeader {
    /// The container metadata (bounds, voxel count, payload location).
    pub meta: GridMeta,
    /// Rows of the affine index→asset map (`pnanovdb_map_apply`):
    /// `asset = mat · index + vec`.
    pub index_to_asset: [[f32; 4]; 3],
    /// Rows of its inverse (`pnanovdb_map_apply_inverse`):
    /// `index = invmat · (asset − vec)`, refolded into one affine row form.
    pub asset_to_index: [[f32; 4]; 3],
    /// What the tree reads outside every tile — and, folded in, the value of
    /// any childless root tile, whose 4096³ span the majorant build covers
    /// conservatively rather than decoding the tile's key. The floor every
    /// majorant cell starts at.
    background: f32,
    /// The upper, lower, and leaf arrays, payload-relative. They are
    /// contiguous and in that order, which is what lets one forward pass
    /// over the payload see every node exactly once.
    nodes: [NodeArray; 3],
}

/// Which of the tree's three node arrays a stretch of payload belongs to.
/// The majorant scan reads each differently: internal nodes contribute the
/// tile values of their childless slots, leaves the maximum of their 512
/// stored voxels.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Level {
    Upper,
    Lower,
    Leaf,
}

/// One level's array of equally sized nodes inside the payload.
#[derive(Clone, Copy, Debug)]
struct NodeArray {
    level: Level,
    start: u64,
    count: u64,
}

impl Level {
    /// Bytes per node of this level, for a float grid (`upper_size`,
    /// `lower_size`, `leaf_size`).
    const fn stride(self) -> u64 {
        match self {
            Self::Upper => 270_400,
            Self::Lower => 33_856,
            Self::Leaf => 2_144,
        }
    }

    /// Byte offset of the node's child mask, and of its value table. A leaf
    /// has no children — the offset is its value mask, which the scan does
    /// not consult: interpolation reads inactive voxels too, so every stored
    /// value counts.
    const fn masks(self) -> (usize, usize) {
        match self {
            Self::Upper => (4128, 8256),
            Self::Lower => (544, 1088),
            Self::Leaf => (16, 96),
        }
    }

    /// How many voxels one node spans per axis, and how many one table slot
    /// does — the two powers of two that turn a slot index back into a
    /// coordinate. Leaves have no slots; their 8³ voxels are the table.
    const fn spans(self) -> (i32, i32) {
        match self {
            Self::Upper => (4096, 128),
            Self::Lower => (128, 8),
            Self::Leaf => (8, 1),
        }
    }
}

impl NodeArray {
    fn end(&self) -> u64 {
        self.start + self.level.stride() * self.count
    }
}

/// Parse [`GridHeader`] for `grid_name` inside the `.nvdb` at `path`,
/// reading only the container metadata and the first bytes of the payload
/// (`GridData` + `TreeData` + the root's stats words).
///
/// # Errors
///
/// [`Error::Scene`] when the container doesn't parse, the named grid is
/// missing or not a float grid, or the payload's magic/type disagree with
/// the container (corrupt file or ABI drift).
pub fn grid_header(path: &Path, grid_name: &str) -> Result<GridHeader> {
    let grids = read_metadata(path)?;
    let meta = grids
        .iter()
        .find(|meta| meta.name == grid_name)
        .ok_or_else(|| {
            let names: Vec<&str> = grids.iter().map(|meta| meta.name.as_str()).collect();
            scene_error(
                path,
                format!(
                    "no grid named \"{grid_name}\" (file holds: {})",
                    names.join(", ")
                ),
            )
        })?
        .clone();
    if meta.grid_type != GRID_TYPE_FLOAT {
        return Err(scene_error(
            path,
            format!(
                "grid \"{grid_name}\" is {}, only Float grids are supported",
                meta.type_name()
            ),
        ));
    }
    let mut file =
        fs::File::open(path).map_err(|e| scene_error(path, format!("can't open: {e}")))?;
    file.seek(SeekFrom::Start(meta.file_offset))
        .map_err(|e| scene_error(path, format!("seek failed: {e}")))?;
    let grid = read_exact::<{ GRID_DATA_SIZE as usize }>(&mut file, path)?;
    let magic = u64::from_le_bytes(grid[0..8].try_into().unwrap());
    if magic != MAGIC_NUMB && magic != MAGIC_GRID {
        return Err(scene_error(
            path,
            format!("grid \"{grid_name}\" payload has no NanoVDB magic ({magic:#018x})"),
        ));
    }
    let payload_type = u32::from_le_bytes(
        grid[GRID_OFF_GRID_TYPE..GRID_OFF_GRID_TYPE + 4]
            .try_into()
            .unwrap(),
    );
    if payload_type != meta.grid_type {
        return Err(scene_error(
            path,
            format!(
                "grid \"{grid_name}\": payload type {payload_type} disagrees with the \
                 container's {} — corrupt file or parser drift",
                meta.grid_type
            ),
        ));
    }
    let (index_to_asset, asset_to_index) = parse_map(&grid);

    let tree = read_exact::<{ TREE_DATA_SIZE as usize }>(&mut file, path)?;
    let (root_start, nodes) = parse_tree(&tree, meta.grid_size).ok_or_else(|| {
        scene_error(
            path,
            format!(
                "grid \"{grid_name}\": the tree's node arrays fall outside the payload's \
                 level order — corrupt file or ABI drift"
            ),
        )
    })?;

    file.seek(SeekFrom::Start(meta.file_offset + root_start))
        .map_err(|e| scene_error(path, format!("seek failed: {e}")))?;
    let root = read_exact::<{ ROOT_SIZE as usize }>(&mut file, path)?;
    let background = f32::from_le_bytes(
        root[ROOT_OFF_BACKGROUND..ROOT_OFF_BACKGROUND + 4]
            .try_into()
            .expect("4 bytes"),
    );
    let tiles = u32::from_le_bytes(
        root[ROOT_OFF_TABLE_SIZE..ROOT_OFF_TABLE_SIZE + 4]
            .try_into()
            .expect("4 bytes"),
    );
    let background = root_floor(&mut file, path, tiles, background)?;
    Ok(GridHeader {
        meta,
        index_to_asset,
        asset_to_index,
        background,
        nodes,
    })
}

/// The root's payload-relative offset and the tree's three node arrays, or
/// `None` if they are not what the one-pass scan requires: the root, then
/// the arrays in level order, none overlapping and all inside the payload.
/// That is the layout `NanoVDB` writes; a file disagreeing with it is one
/// this parser would read as garbage.
fn parse_tree(tree: &[u8; TREE_DATA_SIZE as usize], grid_size: u64) -> Option<(u64, [NodeArray; 3])> {
    let offset = |at: usize| u64::from_le_bytes(tree[at..at + 8].try_into().unwrap());
    let count = |at: usize| u64::from(u32::from_le_bytes(tree[at..at + 4].try_into().unwrap()));
    // Node offsets are relative to the tree, which follows GridData.
    let base = GRID_DATA_SIZE;
    let levels = [
        (Level::Upper, TREE_OFF_NODE_OFFSET_UPPER, TREE_OFF_NODE_COUNT_UPPER),
        (Level::Lower, TREE_OFF_NODE_OFFSET_LOWER, TREE_OFF_NODE_COUNT_LOWER),
        (Level::Leaf, TREE_OFF_NODE_OFFSET_LEAF, TREE_OFF_NODE_COUNT_LEAF),
    ];
    let nodes = levels.map(|(level, at, counted)| NodeArray {
        level,
        start: base + offset(at),
        count: count(counted),
    });
    let root_start = base + offset(TREE_OFF_NODE_OFFSET_ROOT);
    let mut walk = root_start + ROOT_SIZE;
    for array in &nodes {
        if array.start < walk || array.end() > grid_size {
            return None;
        }
        walk = array.end();
    }
    Some((root_start, nodes))
}

/// The value every majorant cell starts at: the background, raised by any
/// childless root tile's value. Such a tile fills a 4096³ block the tree
/// keys by coordinate; rather than decode the key, the floor rises
/// everywhere — the case does not occur in a converted fog volume, and a
/// conservative ceiling costs only tracking efficiency where it does.
///
/// A non-finite floor would poison every cell, so it falls to zero: the
/// shader's density clamp then reads the grid as vacuum rather than as NaN.
fn root_floor(file: &mut fs::File, path: &Path, tiles: u32, background: f32) -> Result<f32> {
    let mut floor = if background.is_finite() { background } else { 0.0 };
    for _ in 0..tiles {
        let tile = read_exact::<{ ROOT_TILE_SIZE as usize }>(file, path)?;
        let child = i64::from_le_bytes(
            tile[ROOT_TILE_OFF_CHILD..ROOT_TILE_OFF_CHILD + 8]
                .try_into()
                .expect("8 bytes"),
        );
        let value = f32::from_le_bytes(
            tile[ROOT_TILE_OFF_VALUE..ROOT_TILE_OFF_VALUE + 4]
                .try_into()
                .expect("4 bytes"),
        );
        if child == 0 && value.is_finite() {
            floor = floor.max(value);
        }
    }
    Ok(floor.max(0.0))
}

/// The grid's affine map out of a `GridData` block: rows of index→asset
/// and — with `index = invmat · (asset − vec)` refolded into one affine —
/// rows of asset→index.
fn parse_map(grid: &[u8; GRID_DATA_SIZE as usize]) -> ([[f32; 4]; 3], [[f32; 4]; 3]) {
    let f32_at =
        |at: usize| f32::from_le_bytes(grid[at..at + 4].try_into().expect("inside GridData"));
    let matf: [f32; 9] = std::array::from_fn(|i| f32_at(GRID_OFF_MAP + MAP_OFF_MATF + 4 * i));
    let invmatf: [f32; 9] =
        std::array::from_fn(|i| f32_at(GRID_OFF_MAP + MAP_OFF_INVMATF + 4 * i));
    let vecf: [f32; 3] = std::array::from_fn(|i| f32_at(GRID_OFF_MAP + MAP_OFF_VECF + 4 * i));
    let index_to_asset: [[f32; 4]; 3] = std::array::from_fn(|row| {
        [
            matf[3 * row],
            matf[3 * row + 1],
            matf[3 * row + 2],
            vecf[row],
        ]
    });
    let asset_to_index: [[f32; 4]; 3] = std::array::from_fn(|row| {
        [
            invmatf[3 * row],
            invmatf[3 * row + 1],
            invmatf[3 * row + 2],
            -(invmatf[3 * row] * vecf[0]
                + invmatf[3 * row + 1] * vecf[1]
                + invmatf[3 * row + 2] * vecf[2]),
        ]
    });
    (index_to_asset, asset_to_index)
}

/// The shell's box in index space: the grid's active bounds (inclusive)
/// dilated one voxel each way, which is exactly the trilinear stencil's
/// support — a sample at index `p` reads the lattice from `floor(p)` to
/// `floor(p) + 1`, so density is zero outside this box. The `MediumBounds`
/// unit cube maps onto it, and so does the majorant lattice.
#[must_use]
pub fn shell_box(index_bbox: &[i32; 6]) -> ([f32; 3], [f32; 3]) {
    (
        std::array::from_fn(|axis| index_bbox[axis] as f32 - 1.0),
        std::array::from_fn(|axis| index_bbox[axis + 3] as f32 + 1.0),
    )
}

/// Cells along the shell's longest axis. Measured over `bunny_cloud`,
/// explosion, and `wdas_cloud`: the tracker's cost is flat within 10 % from
/// 32 to 128 and rises steeply below 16, so the choice is the middle of a
/// plateau rather than a peak. 1 collapses the lattice to the single global
/// majorant — the scalar tracker this replaced, bit for bit, which is how
/// the walk is checked against it.
const MAJORANT_RESOLUTION: u32 = 64;

/// The smallest a cell may be, in voxels. The lattice is built from one
/// maximum per `NanoVDB` leaf — an 8³ block — so a finer cell would repeat
/// the same ceiling across more cells: more `DDA` steps, no tighter bound.
const MAJORANT_MIN_CELL: f32 = 8.0;

/// [`MAJORANT_RESOLUTION`], overridable for the resolution sweep and for the
/// `1` that renders as the scalar tracker. A measurement hook, deliberately
/// not a render setting: the resolution is a property of how the tracker is
/// built, not something a scene has an opinion about.
fn majorant_target() -> u32 {
    static TARGET: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| {
        std::env::var("CENOTE_MAJORANT_RES")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(MAJORANT_RESOLUTION)
            .max(1)
    });
    *TARGET
}

/// The majorant lattice's cell counts for a grid with these active bounds:
/// cubic cells (in voxels) sized so the longest axis gets
/// [`majorant_target`] of them, never finer than [`MAJORANT_MIN_CELL`].
/// A pure function of the grid, so the medium's transform and the pool's
/// lattice cannot disagree about the resolution.
#[must_use]
pub fn majorant_resolution(index_bbox: &[i32; 6]) -> [u32; 3] {
    let (lo, hi) = shell_box(index_bbox);
    let extent: [f32; 3] = std::array::from_fn(|axis| hi[axis] - lo[axis]);
    let longest = extent.iter().copied().fold(0.0f32, f32::max);
    let cell = (longest / majorant_target() as f32).max(MAJORANT_MIN_CELL);
    extent.map(|axis| ((axis / cell).ceil() as u32).max(1))
}

/// A grid's majorant lattice: one unitless density ceiling per cell over the
/// shell box, the bound the tracker's free flights are drawn against. Built
/// from the tree in one forward pass — every leaf's largest stored voxel and
/// every childless tile's value, each splatted over the cells its own
/// trilinear support can reach.
///
/// The maxima come from the *stored* values, not from `NanoVDB`'s per-node
/// statistics: interpolation reads inactive voxels too, and a `.nvdb` whose
/// stats are absent or stale would otherwise underestimate the bound.
struct Majorants {
    /// Cells per axis.
    res: [u32; 3],
    /// `res.x · res.y · res.z` ceilings, x fastest — the order the shader's
    /// cell lookup assumes.
    cells: Vec<f32>,
    /// Index-space position of cell (0,0,0)'s corner, and cells per index
    /// unit: the splat's own mapping, and the one the record carries.
    lo: [f32; 3],
    scale: [f32; 3],
    /// Whether any density read was `NaN` or `±∞`. Reported once per grid
    /// rather than silently tolerated: such a voxel never becomes a ceiling,
    /// so the shader's density bound reads it as whatever finite ceiling
    /// surrounds it.
    malformed: bool,
}

impl Majorants {
    /// An empty lattice, every cell at `floor` — what the tree reads
    /// wherever no node covers it.
    fn new(index_bbox: &[i32; 6], floor: f32) -> Self {
        let res = majorant_resolution(index_bbox);
        let (lo, hi) = shell_box(index_bbox);
        Self {
            res,
            cells: vec![floor; (res[0] * res[1] * res[2]) as usize],
            lo,
            scale: std::array::from_fn(|axis| res[axis] as f32 / (hi[axis] - lo[axis])),
            malformed: false,
        }
    }

    /// Raise every cell a lattice value in `[min, max]` (inclusive, index
    /// space) can reach to at least `value`. The reach is the block dilated
    /// one voxel: a position `p` interpolates the lattice points `floor(p)`
    /// and `floor(p) + 1`, so a value at `v` is read anywhere in
    /// `[v − 1, v + 1]`.
    ///
    /// The one gate every contribution passes: below `floor` it changes
    /// nothing, and NaN or ±∞ is dropped rather than splatted — an infinite
    /// ceiling would make the tracker's majorant infinite, where dropping it
    /// leaves the shader's density bound reading that voxel as the finite
    /// ceiling around it.
    fn raise(&mut self, min: [i32; 3], max: [i32; 3], value: f32, floor: f32) {
        if !value.is_finite() {
            self.malformed = true;
            return;
        }
        if value <= floor {
            return;
        }
        let cell = |axis: usize, index: f32| {
            (((index - self.lo[axis]) * self.scale[axis]).floor())
                .clamp(0.0, (self.res[axis] - 1) as f32) as u32
        };
        let from: [u32; 3] = std::array::from_fn(|axis| cell(axis, min[axis] as f32 - 1.0));
        let to: [u32; 3] = std::array::from_fn(|axis| cell(axis, max[axis] as f32 + 1.0));
        for z in from[2]..=to[2] {
            for y in from[1]..=to[1] {
                let row = ((z * self.res[1] + y) * self.res[0]) as usize;
                for cell in &mut self.cells[row + from[0] as usize..=row + to[0] as usize] {
                    *cell = cell.max(value);
                }
            }
        }
    }

    /// Fold in every node of one level packed back to back.
    fn scan_all(&mut self, nodes: &[u8], level: Level, floor: f32) {
        for node in nodes.chunks_exact(level.stride() as usize) {
            self.scan(node, level, floor);
        }
    }

    /// Fold one node's contribution in. `node` is the whole node, `floor`
    /// the value below which a contribution changes nothing.
    fn scan(&mut self, node: &[u8], level: Level, floor: f32) {
        let (node_span, slot_span) = level.spans();
        let origin: [i32; 3] = std::array::from_fn(|axis| {
            i32::from_le_bytes(node[4 * axis..4 * axis + 4].try_into().expect("4 bytes"))
                & !(node_span - 1)
        });
        let (child_mask, table) = level.masks();
        if level == Level::Leaf {
            // Every stored voxel, active or not: the accessor reads them
            // all. Written as a comparison rather than `f32::max` so it
            // vectorizes — `max`'s NaN rule does not — which matters at a
            // billion voxels; a NaN loses the comparison either way.
            let voxels = || {
                node[table..table + 4 * 512]
                    .chunks_exact(4)
                    .map(|value| f32::from_le_bytes(value.try_into().expect("4 bytes")))
            };
            let mut max = f32::NEG_INFINITY;
            for value in voxels() {
                if value > max {
                    max = value;
                }
            }
            if !max.is_finite() {
                // A single ±∞ voxel must not cost its 511 neighbours their
                // ceiling. Off the hot path by construction: reaching here
                // means the leaf holds no finite value above its own maximum.
                self.malformed = true;
                max = voxels()
                    .filter(|value| value.is_finite())
                    .fold(f32::NEG_INFINITY, f32::max);
            }
            self.raise(origin, origin.map(|axis| axis + node_span - 1), max, floor);
            return;
        }
        // A slot with a child is covered by that child's own scan; one
        // without holds the value the accessor reads across its whole span,
        // whether or not the tile is active.
        let slots = (node_span / slot_span) as u32;
        let bits = slots.trailing_zeros();
        for slot in 0..slots * slots * slots {
            let word = child_mask + 4 * (slot as usize >> 5);
            let mask = u32::from_le_bytes(node[word..word + 4].try_into().expect("4 bytes"));
            if mask >> (slot & 31) & 1 != 0 {
                continue;
            }
            let at = table + 8 * slot as usize;
            let value = f32::from_le_bytes(node[at..at + 4].try_into().expect("4 bytes"));
            let min: [i32; 3] = std::array::from_fn(|axis| {
                let shift = bits * (2 - axis as u32);
                let along = i32::try_from((slot >> shift) & (slots - 1)).expect("five bits");
                origin[axis] + along * slot_span
            });
            self.raise(min, min.map(|axis| axis + slot_span - 1), value, floor);
        }
    }

    /// Fold `other`'s ceilings into these, cell by cell — how the parallel
    /// scan's per-thread lattices become one. Both cover the same shell at
    /// the same resolution, [`Majorants::new`] being a pure function of the
    /// grid.
    fn merge(&mut self, other: &Self) {
        for (cell, &raised) in self.cells.iter_mut().zip(&other.cells) {
            *cell = cell.max(raised);
        }
        self.malformed |= other.malformed;
    }
}

/// Bumping this invalidates every prep cache — the knob to turn when the
/// conversion (stats mode, codec, grid selection) changes meaning.
const PREP_VERSION: u32 = 1;

/// Resolve a volume reference to the `.nvdb` the renderer reads: `.nvdb`
/// passes through untouched, `.vdb` goes through the beside-the-source
/// cache (`cloud.vdb` → `cloud.vdb.nvdb`), converting via `cenote-vdb-prep`
/// on a miss.
///
/// Cache validity is content-hashed like the DDS cache: the sidecar
/// (`cloud.vdb.nvdb.src`) records the source's `FNV-1a` hash with its size
/// and mtime, so an unchanged file revalidates without rehashing
/// multi-GiB bytes, a touched one rehashes and stays a hit, and an edited
/// one re-preps.
///
/// # Errors
///
/// [`Error::Scene`] when the source is missing, the converter isn't
/// available (the message says how to build it), or the conversion fails.
pub fn prepared(path: &Path) -> Result<PathBuf> {
    if path.extension().is_some_and(|ext| ext == "nvdb") {
        return Ok(path.to_owned());
    }
    let stat = fs::metadata(path).map_err(|e| scene_error(path, format!("can't stat: {e}")))?;
    let mtime = stat
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos());
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".nvdb");
    let cache = path.with_file_name(name);
    let mut sidecar_name = cache.file_name().unwrap_or_default().to_os_string();
    sidecar_name.push(".src");
    let sidecar = cache.with_file_name(sidecar_name);

    let recorded = fs::read_to_string(&sidecar).ok().and_then(|s| {
        let mut parts = s.split_whitespace();
        Some((
            parts.next()?.parse::<u64>().ok()?,
            parts.next()?.parse::<u128>().ok()?,
            parts.next()?.parse::<u64>().ok()?,
        ))
    });
    if cache.is_file()
        && let Some((size, recorded_mtime, hash)) = recorded
        && size == stat.len()
    {
        if recorded_mtime == mtime {
            return Ok(cache);
        }
        // The mtime moved: the content decides. A touched-but-equal source
        // stays a hit and re-records the new mtime.
        if source_hash(path)? == hash {
            let _ = fs::write(&sidecar, format!("{} {mtime} {hash}", stat.len()));
            return Ok(cache);
        }
    }

    let hash = source_hash(path)?;
    let tool = find_prep_tool().ok_or_else(|| {
        scene_error(
            path,
            "cenote-vdb-prep not found — build it (cmake -S vdb-prep -B build/vdb-prep && \
             cmake --build build/vdb-prep) and put it on PATH, or set CENOTE_VDB_PREP",
        )
    })?;
    log::info!("prepping {} → {}", path.display(), cache.display());
    // Convert to a temp name, then rename: a crash mid-write can't leave a
    // plausible-looking cache behind.
    let mut partial_name = cache.file_name().unwrap_or_default().to_os_string();
    partial_name.push(".partial");
    let partial = cache.with_file_name(partial_name);
    let output = std::process::Command::new(&tool)
        .arg(path)
        .arg(&partial)
        .output()
        .map_err(|e| scene_error(path, format!("can't run {}: {e}", tool.display())))?;
    if !output.status.success() {
        let _ = fs::remove_file(&partial);
        return Err(scene_error(
            path,
            format!(
                "cenote-vdb-prep failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            ),
        ));
    }
    fs::rename(&partial, &cache)
        .map_err(|e| scene_error(path, format!("can't move prep output into place: {e}")))?;
    if let Err(e) = fs::write(&sidecar, format!("{} {mtime} {hash}", stat.len())) {
        log::warn!(
            "volume \"{}\": couldn't write its cache sidecar: {e}",
            path.display()
        );
    }
    Ok(cache)
}

/// `FNV-1a` over the source file (streamed — sources are multi-GiB) and the
/// prep parameters, continuing the DDS cache's recipe.
fn source_hash(path: &Path) -> Result<u64> {
    let mut file =
        fs::File::open(path).map_err(|e| scene_error(path, format!("can't read: {e}")))?;
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut chunk = vec![0u8; 1 << 20];
    loop {
        let n = file
            .read(&mut chunk)
            .map_err(|e| scene_error(path, format!("can't read: {e}")))?;
        if n == 0 {
            break;
        }
        hash = fnv1a(hash, &chunk[..n]);
    }
    Ok(fnv1a(hash, &PREP_VERSION.to_le_bytes()))
}

/// `FNV-1a` over `bytes`, continuing from `state` — the DDS cache's hash,
/// with the same non-cryptographic contract: a collision costs one stale
/// cache the user can delete.
fn fnv1a(state: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(state, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// Locate `cenote-vdb-prep`: an explicit `CENOTE_VDB_PREP` wins, then
/// PATH, then — as a source-checkout convenience, like hot reload's baked
/// shader paths — the conventional build tree next to this crate.
/// `pub(crate)` for the grid gates in `render/tests.rs`, which synthesize
/// their fixture grids through the tool's `--constant` mode and skip —
/// like GPU-less machines skip GPU tests — where it isn't built.
pub(crate) fn find_prep_tool() -> Option<PathBuf> {
    if let Some(tool) = std::env::var_os("CENOTE_VDB_PREP") {
        let tool = PathBuf::from(tool);
        return tool.is_file().then_some(tool);
    }
    let on_path = |name: &str| {
        std::env::var_os("PATH").and_then(|paths| {
            std::env::split_paths(&paths)
                .map(|dir| dir.join(name))
                .find(|candidate| candidate.is_file())
        })
    };
    on_path("cenote-vdb-prep").or_else(|| {
        let checkout = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/vdb-prep/cenote-vdb-prep");
        checkout.is_file().then_some(checkout)
    })
}

/// Chunk size below which the scan stays on one thread: spawning is only
/// worth its per-thread lattice allocation for a chunk that takes real time.
const SCAN_PARALLEL_FROM: usize = 16 << 20;

/// Fold a chunk of whole nodes into `majorants`. The reduction touches every
/// stored voxel — gigabytes on a production grid, and the largest single
/// cost of making one resident — while a single thread runs it at roughly
/// one core's memory bandwidth. So each thread raises a lattice of its own
/// (hundreds of KiB) over its share of the nodes and the maxima merge, which
/// is order-free and therefore gives the same lattice at any thread count.
fn scan_nodes(majorants: &mut Majorants, chunk: &[u8], level: Level, header: &GridHeader) {
    let (floor, bbox) = (header.background, &header.meta.index_bbox);
    let threads = (chunk.len() / SCAN_PARALLEL_FROM)
        .min(std::thread::available_parallelism().map_or(1, std::num::NonZero::get))
        .max(1);
    if threads == 1 {
        majorants.scan_all(chunk, level, floor);
        return;
    }
    let stride = level.stride() as usize;
    let per = (chunk.len() / stride).div_ceil(threads) * stride;
    std::thread::scope(|scope| {
        let parts: Vec<_> = chunk
            .chunks(per)
            .map(|part| {
                scope.spawn(move || {
                    let mut local = Majorants::new(bbox, floor);
                    local.scan_all(part, level, floor);
                    local
                })
            })
            .collect();
        for part in parts {
            majorants.merge(&part.join().expect("the majorant scan panicked"));
        }
    });
}

/// Where a resident grid's two arrays sit in the pool — the pair a
/// heterogeneous `MediumRecord` carries.
#[derive(Clone, Copy, Debug)]
pub struct ResidentGrid {
    /// Byte offset of the `NanoVDB` payload: a `pnanovdb_grid_handle_t`.
    pub grid: u32,
    /// Byte offset of the majorant lattice, `res.x·res.y·res.z` floats laid
    /// out x fastest. Its resolution is [`majorant_resolution`] of the same
    /// grid — recomputed rather than carried, so the two cannot disagree.
    pub majorant: u32,
    /// The largest ceiling in that lattice, which is therefore the whole
    /// grid's: what bounds it where the tracker walks another grid's.
    pub density_max: f32,
}

/// The GPU home of every resident `NanoVDB` grid: one grow-only
/// device-local buffer, grids appended at 32-byte-aligned offsets, keyed
/// by (canonical file, grid name) so a grid shared across media and
/// instances uploads once. Eviction is scene-swap-scoped: drop the pool.
pub struct GridPool {
    /// Lazily created — Vulkan forbids zero-sized buffers and an empty
    /// pool is the common case (every scene without a volume grid).
    buffer: Option<Buffer>,
    /// First free byte.
    cursor: u64,
    /// Where each resident grid landed, keyed by (canonical file, name).
    resident: HashMap<(PathBuf, String), ResidentGrid>,
}

impl GridPool {
    /// `PNanoVDB` addresses are 32-bit byte offsets — the pool's hard cap.
    pub const CAPACITY: u64 = 1 << 32;

    /// Staging-buffer size for streamed uploads: large enough that a
    /// multi-GiB grid moves in a few submits, small enough that host
    /// memory never holds more than this of the file at once.
    const STAGING_CHUNK: u64 = 256 << 20;

    /// An empty pool. Allocates nothing until the first grid arrives.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffer: None,
            cursor: 0,
            resident: HashMap::new(),
        }
    }

    /// The pool buffer for descriptor binding — `None` until a grid has
    /// been uploaded.
    #[must_use]
    pub fn buffer(&self) -> Option<&Buffer> {
        self.buffer.as_ref()
    }

    /// Make `grid_name` from the `.nvdb` at `path` resident — payload and
    /// majorant lattice both — and return where the two landed. A grid
    /// already resident under the same canonical path and name returns its
    /// offsets without touching the file.
    ///
    /// # Errors
    ///
    /// [`Error::Scene`] when the file doesn't parse, the named grid is
    /// missing or not a float grid, or residence would exceed the 4 GiB
    /// the pool can address; GPU errors from allocation or upload.
    pub fn upload(&mut self, gpu: &Context, path: &Path, grid_name: &str) -> Result<ResidentGrid> {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_owned());
        let key = (canonical, grid_name.to_owned());
        if let Some(&resident) = self.resident.get(&key) {
            return Ok(resident);
        }
        let header = grid_header(path, grid_name)?;
        let res = majorant_resolution(&header.meta.index_bbox);
        let lattice_bytes = 4 * u64::from(res[0]) * u64::from(res[1]) * u64::from(res[2]);
        let grid = self.cursor.next_multiple_of(DATA_ALIGNMENT);
        let majorant = (grid + header.meta.grid_size).next_multiple_of(DATA_ALIGNMENT);
        let end = majorant
            .checked_add(lattice_bytes)
            .filter(|&end| end <= Self::CAPACITY)
            .ok_or_else(|| {
                scene_error(
                    path,
                    format!(
                        "grid \"{grid_name}\" ({} bytes) does not fit the pool's 4 GiB address \
                         space ({} bytes already resident)",
                        header.meta.grid_size,
                        self.cursor
                    ),
                )
            })?;
        // One growth for both arrays: `reserve` copies everything resident,
        // and doing it twice per grid would copy the pool twice.
        self.reserve(gpu, end)?;
        let majorants = self.stream_payload(gpu, path, &header, grid)?;
        if majorants.malformed {
            log::warn!(
                "volume \"{}\": grid \"{grid_name}\" holds NaN or infinite densities; \
                 they bound to the finite ceiling around them rather than to themselves",
                path.display()
            );
        }
        self.upload_bytes(gpu, bytemuck::cast_slice(&majorants.cells), majorant)?;
        self.cursor = end;
        let resident = ResidentGrid {
            grid: u32::try_from(grid).expect("offset fits: end <= CAPACITY"),
            majorant: u32::try_from(majorant).expect("offset fits: end <= CAPACITY"),
            density_max: majorants.cells.iter().copied().fold(0.0, f32::max),
        };
        log::debug!(
            "volume \"{}\": grid \"{grid_name}\" {} MiB, majorant {}×{}×{} ({} KiB)",
            path.display(),
            header.meta.grid_size >> 20,
            res[0],
            res[1],
            res[2],
            lattice_bytes >> 10
        );
        self.resident.insert(key, resident);
        Ok(resident)
    }

    /// Grow the pool to hold at least `needed` bytes, copying residents
    /// into the replacement. Exact-fit growth: grids arrive once per scene
    /// load and the 4 GiB ceiling leaves no room for doubling headroom.
    fn reserve(&mut self, gpu: &Context, needed: u64) -> Result<()> {
        if self.buffer.as_ref().is_some_and(|b| b.size() >= needed) {
            return Ok(());
        }
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::TRANSFER_SRC;
        let grown = gpu.create_buffer("vdb.pool", needed, usage, MemoryLocation::GpuOnly)?;
        if let Some(old) = self.buffer.take()
            && self.cursor > 0
        {
            gpu.copy_buffers(&[(&old, &grown, self.cursor)])?;
        }
        self.buffer = Some(grown);
        Ok(())
    }

    /// Stream a grid's payload from the file into the pool at `offset`,
    /// building its majorant lattice from the same bytes on the way past.
    /// Host memory holds at most [`Self::STAGING_CHUNK`] of a multi-GiB grid
    /// at once.
    ///
    /// The file reads straight into the staging buffer and the scan reads it
    /// back in place, which is why staging is [`MemoryLocation::GpuToCpu`]
    /// (host-*cached*) and not `CpuToGpu` (host-visible device memory, where
    /// a read runs at `PCIe` speed with no cache to amortize it).
    ///
    /// Chunks never straddle a node array's boundary and, inside one, always
    /// hold whole nodes — which is what lets the scan address a node by
    /// index instead of stitching one across two reads.
    fn stream_payload(
        &mut self,
        gpu: &Context,
        path: &Path,
        header: &GridHeader,
        offset: u64,
    ) -> Result<Majorants> {
        let meta = &header.meta;
        let mut majorants = Majorants::new(&meta.index_bbox, header.background);
        let pool = self.buffer.as_ref().expect("reserved before streaming");
        let mut file =
            fs::File::open(path).map_err(|e| scene_error(path, format!("can't open: {e}")))?;
        file.seek(SeekFrom::Start(meta.file_offset))
            .map_err(|e| scene_error(path, format!("seek failed: {e}")))?;
        let chunk_bytes = meta.grid_size.min(Self::STAGING_CHUNK);
        let mut staging = gpu.create_buffer(
            "vdb.staging",
            chunk_bytes,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::GpuToCpu,
        )?;
        let mut sent = 0u64;
        while sent < meta.grid_size {
            let array = header.nodes.iter().find(|a| (a.start..a.end()).contains(&sent));
            let limit = header
                .nodes
                .iter()
                .flat_map(|a| [a.start, a.end()])
                .chain([meta.grid_size])
                .filter(|&cut| cut > sent)
                .min()
                .expect("grid_size is a cut past every offset");
            let mut bytes = chunk_bytes.min(limit - sent);
            if let Some(array) = array {
                bytes -= bytes % array.level.stride();
            }
            // `sent` sits on a node boundary inside an array — the cuts put
            // it there and whole nodes keep it there — so a chunk always
            // holds at least one, and the loop always advances.
            assert!(bytes > 0, "a node array shorter than one node");
            let chunk = &mut staging.mapped_mut()[..bytes as usize];
            file.read_exact(chunk)
                .map_err(|e| scene_error(path, format!("truncated payload: {e}")))?;
            if sent == 0 {
                // The payload opens with GridData, whose magic double-checks
                // the container parse actually landed on a grid.
                let magic = u64::from_le_bytes(chunk[..8].try_into().unwrap());
                if magic != MAGIC_NUMB && magic != MAGIC_GRID {
                    return Err(scene_error(
                        path,
                        format!(
                            "grid \"{}\" payload has no NanoVDB magic (found {magic:#018x}) — \
                             corrupt file or parser drift",
                            meta.name
                        ),
                    ));
                }
            }
            if let Some(array) = array {
                scan_nodes(
                    &mut majorants,
                    &staging.mapped()[..bytes as usize],
                    array.level,
                    header,
                );
            }
            gpu.copy_buffer_region(&staging, pool, 0, offset + sent, bytes)?;
            sent += bytes;
        }
        Ok(majorants)
    }

    /// Copy `bytes` into the pool at `offset` through a one-shot staging
    /// buffer — the majorant lattice, which is megabytes at most.
    fn upload_bytes(&self, gpu: &Context, bytes: &[u8], offset: u64) -> Result<()> {
        let pool = self.buffer.as_ref().expect("reserved before uploading");
        let mut staging = gpu.create_buffer(
            "vdb.majorant",
            bytes.len() as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        )?;
        staging.mapped_mut()[..bytes.len()].copy_from_slice(bytes);
        gpu.copy_buffer_region(&staging, pool, 0, offset, bytes.len() as u64)
    }
}

impl Default for GridPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialize a minimal single-segment `.nvdb` holding the named grids
    /// with the given payloads — enough container for the parser, no
    /// actual tree.
    fn synthesize(grids: &[(&str, u32, &[u8])]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC_NUMB.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes()); // version
        out.extend_from_slice(&(grids.len() as u16).to_le_bytes());
        out.extend_from_slice(&0u16.to_le_bytes()); // codec NONE
        for &(name, grid_type, payload) in grids {
            let mut meta = [0u8; 176];
            meta[0..8].copy_from_slice(&(payload.len() as u64).to_le_bytes()); // gridSize
            meta[8..16].copy_from_slice(&(payload.len() as u64).to_le_bytes()); // fileSize
            meta[32..36].copy_from_slice(&grid_type.to_le_bytes());
            let name_bytes = name.len() as u32 + 1; // with NUL
            meta[136..140].copy_from_slice(&name_bytes.to_le_bytes());
            out.extend_from_slice(&meta);
            out.extend_from_slice(name.as_bytes());
            out.push(0);
        }
        for &(_, _, payload) in grids {
            out.extend_from_slice(payload);
        }
        out
    }

    fn temp_nvdb(name: &str, bytes: &[u8]) -> PathBuf {
        let path = std::env::temp_dir().join(format!("cenote-vdb-{}-{name}", std::process::id()));
        fs::write(&path, bytes).expect("write temp nvdb");
        path
    }

    #[test]
    fn metadata_round_trips() {
        let payload_a = vec![1u8; 64];
        let payload_b = vec![2u8; 96];
        let bytes = synthesize(&[
            ("density", GRID_TYPE_FLOAT, &payload_a),
            ("temperature", GRID_TYPE_FLOAT, &payload_b),
        ]);
        let path = temp_nvdb("roundtrip.nvdb", &bytes);
        let grids = read_metadata(&path).expect("parse");
        let _ = fs::remove_file(&path);
        assert_eq!(grids.len(), 2);
        assert_eq!(grids[0].name, "density");
        assert_eq!(grids[0].grid_size, 64);
        assert_eq!(grids[1].name, "temperature");
        // Payloads follow the header (16), two metas (176 each) and names
        // (8 + 12 with NULs).
        assert_eq!(grids[0].file_offset, 16 + 176 + 8 + 176 + 12);
        assert_eq!(grids[1].file_offset, grids[0].file_offset + 64);
        assert_eq!(grids[1].file_offset + 96, bytes.len() as u64);
    }

    /// A payload with a real-enough `GridData`/`TreeData`/root prefix: the
    /// magic, the grid type, a scale-and-translate map, and an empty root —
    /// what [`grid_header`] parses, laid out at the ABI 32.7 offsets. The
    /// three node arrays are empty and start where the root ends.
    fn grid_payload(scale: f32, translate: [f32; 3], background: f32) -> Vec<u8> {
        let mut payload = vec![0u8; 800]; // GridData 672 + TreeData 64 + root 64
        payload[0..8].copy_from_slice(&MAGIC_NUMB.to_le_bytes());
        payload[GRID_OFF_GRID_TYPE..GRID_OFF_GRID_TYPE + 4]
            .copy_from_slice(&GRID_TYPE_FLOAT.to_le_bytes());
        let mut write_f32 = |at: usize, value: f32| {
            payload[at..at + 4].copy_from_slice(&value.to_le_bytes());
        };
        for (row, &offset) in translate.iter().enumerate() {
            write_f32(GRID_OFF_MAP + MAP_OFF_MATF + 4 * (3 * row + row), scale);
            write_f32(
                GRID_OFF_MAP + MAP_OFF_INVMATF + 4 * (3 * row + row),
                1.0 / scale,
            );
            write_f32(GRID_OFF_MAP + MAP_OFF_VECF + 4 * row, offset);
        }
        write_f32(736 + ROOT_OFF_BACKGROUND, background);
        // Offsets are tree-relative: the root directly after the 64-byte
        // tree, the (empty) node arrays directly after the root.
        let tree = GRID_DATA_SIZE as usize;
        for (at, value) in [
            (TREE_OFF_NODE_OFFSET_ROOT, TREE_DATA_SIZE),
            (TREE_OFF_NODE_OFFSET_UPPER, TREE_DATA_SIZE + ROOT_SIZE),
            (TREE_OFF_NODE_OFFSET_LOWER, TREE_DATA_SIZE + ROOT_SIZE),
            (TREE_OFF_NODE_OFFSET_LEAF, TREE_DATA_SIZE + ROOT_SIZE),
        ] {
            payload[tree + at..tree + at + 8].copy_from_slice(&value.to_le_bytes());
        }
        payload
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "the parser copies bytes; the values must come back bit-exact"
    )]
    fn grid_header_parses_map_and_background() {
        let payload = grid_payload(0.5, [1.0, 2.0, 3.0], 4.5);
        let bytes = synthesize(&[("density", GRID_TYPE_FLOAT, &payload)]);
        let path = temp_nvdb("header.nvdb", &bytes);
        let header = grid_header(&path, "density").expect("parse header");
        let _ = fs::remove_file(&path);
        assert_eq!(header.background, 4.5);
        // index (1,1,1) → asset (1.5, 2.5, 3.5) and back.
        let apply = |rows: &[[f32; 4]; 3], p: [f32; 3]| {
            std::array::from_fn::<f32, 3, _>(|r| {
                rows[r][0] * p[0] + rows[r][1] * p[1] + rows[r][2] * p[2] + rows[r][3]
            })
        };
        let asset = apply(&header.index_to_asset, [1.0, 1.0, 1.0]);
        assert_eq!(asset, [1.5, 2.5, 3.5]);
        assert_eq!(apply(&header.asset_to_index, asset), [1.0, 1.0, 1.0]);
    }

    /// The scan's node arithmetic against the vendored header: a drift in
    /// `PNanoVDB.h`'s float layout would have the scan reading the wrong
    /// words and silently *under*estimating the majorant, which is the one
    /// error the tracker cannot defend against.
    #[test]
    fn float_layout_matches_pnanovdb() {
        let header = fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders/PNanoVDB.h"),
        )
        .expect("the vendored header");
        let define = |name: &str| -> u64 {
            header
                .lines()
                .find_map(|line| line.strip_prefix(&format!("#define {name} ")))
                .unwrap_or_else(|| panic!("{name} is defined"))
                .trim()
                .parse()
                .expect("a plain integer")
        };
        // The float row of pnanovdb_grid_type_constants — the second table
        // entry, one line of comma-separated offsets.
        let float_row: Vec<u64> = header
            .lines()
            .skip_while(|line| !line.contains("pnanovdb_grid_type_constants[PNANOVDB_GRID_TYPE_END]"))
            .filter(|line| line.starts_with('{') && line.len() > 2)
            .nth(1)
            .expect("the Float row, after Unknown's")
            .trim_matches(|c: char| c == '{' || c == '}' || c == ',')
            .split(',')
            .map(|field| field.trim().parse().expect("a plain integer"))
            .collect();
        // upper_off_table, upper_size, lower_off_table, lower_size,
        // leaf_off_table, leaf_size — indices into the row's field order.
        for (level, (table, size)) in [
            (Level::Upper, (float_row[14], float_row[15])),
            (Level::Lower, (float_row[20], float_row[21])),
            (Level::Leaf, (float_row[26], float_row[27])),
        ] {
            assert_eq!(level.stride(), size, "{level:?} stride");
            assert_eq!(level.masks().1 as u64, table, "{level:?} table");
        }
        assert_eq!(Level::Upper.masks().0 as u64, define("PNANOVDB_UPPER_OFF_CHILD_MASK"));
        assert_eq!(Level::Lower.masks().0 as u64, define("PNANOVDB_LOWER_OFF_CHILD_MASK"));
        assert_eq!(Level::Leaf.masks().0 as u64, define("PNANOVDB_LEAF_OFF_VALUE_MASK"));
        // Both node origins come from a bbox min at offset zero.
        assert_eq!(define("PNANOVDB_UPPER_OFF_BBOX_MIN"), 0);
        assert_eq!(define("PNANOVDB_LEAF_OFF_BBOX_MIN"), 0);
        assert_eq!(GRID_DATA_SIZE, define("PNANOVDB_GRID_SIZE"));
        assert_eq!(TREE_DATA_SIZE, define("PNANOVDB_TREE_SIZE"));
        assert_eq!(ROOT_SIZE, float_row[5], "root_size");
        assert_eq!(ROOT_OFF_BACKGROUND as u64, float_row[0], "root_off_background");
        assert_eq!(ROOT_TILE_SIZE, float_row[9], "root_tile_size");
        assert_eq!(ROOT_TILE_OFF_VALUE as u64, float_row[8], "root_tile_off_value");
        assert_eq!(ROOT_OFF_TABLE_SIZE as u64, define("PNANOVDB_ROOT_OFF_TABLE_SIZE"));
        // The scan indexes internal tables by `8 * slot`.
        assert_eq!(float_row[7], 8, "table_stride");
    }

    /// The lattice's two jobs: a leaf's largest stored voxel becomes the
    /// ceiling of every cell its trilinear support can reach, and a
    /// childless tile's value covers its own span.
    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "the ceilings are splatted verbatim; a near-miss would be a bug"
    )]
    fn majorants_bound_leaves_and_tiles() {
        fn at(majorants: &Majorants, x: u32, y: u32, z: u32) -> f32 {
            majorants.cells[((z * 9 + y) * 9 + x) as usize]
        }
        // Active [0, 63]³ → a 65³ shell; at one cell per 8 voxels, 9³ cells.
        let bbox = [0, 0, 0, 63, 63, 63];
        let mut majorants = Majorants::new(&bbox, 0.25);
        assert_eq!(majorants.res, [9, 9, 9]);
        assert!(majorants.cells.iter().all(|&cell| cell == 0.25));

        // A leaf at the origin, one voxel raised. Its 8³ block dilated by
        // one voxel spans index [−1, 8] — cells 0 and 1 on every axis.
        let mut leaf = vec![0u8; Level::Leaf.stride() as usize];
        leaf[Level::Leaf.masks().1 + 4 * 9..Level::Leaf.masks().1 + 4 * 9 + 4]
            .copy_from_slice(&2.0f32.to_le_bytes());
        majorants.scan(&leaf, Level::Leaf, 0.25);
        assert_eq!(at(&majorants, 0, 0, 0), 2.0);
        assert_eq!(at(&majorants, 1, 1, 1), 2.0);
        assert_eq!(at(&majorants, 2, 0, 0), 0.25);

        // A lower node at the origin whose slot (1, 0, 0) is a childless
        // tile: 8 voxels wide, index [8, 15] dilated to [7, 16].
        let (child_mask, table) = Level::Lower.masks();
        let mut lower = vec![0u8; Level::Lower.stride() as usize];
        let slot = 1 << 8; // ((x & 127) >> 3) << 8
        lower[table + 8 * slot..table + 8 * slot + 4].copy_from_slice(&3.0f32.to_le_bytes());
        // Slot 0 holds a far larger value behind a child: the child's own
        // leaf is the authority there, so this must never reach the lattice.
        lower[table..table + 4].copy_from_slice(&9.0f32.to_le_bytes());
        lower[child_mask..child_mask + 4].copy_from_slice(&1u32.to_le_bytes());
        majorants.scan(&lower, Level::Lower, 0.25);
        assert_eq!(at(&majorants, 1, 0, 0), 3.0);
        assert_eq!(at(&majorants, 2, 0, 0), 3.0);
        assert_eq!(at(&majorants, 3, 0, 0), 0.25);
        assert_eq!(at(&majorants, 0, 0, 0), 2.0);
    }

    /// A tree with a bit of everything: 2³ leaves of pseudo-random voxels
    /// under one lower node, a childless tile beside them, and the
    /// background around the lot. Returns the lattice built from it and the
    /// field it was built from, as a closure over integer coordinates —
    /// what `pnanovdb_readaccessor_get_value` would return.
    fn tree_fixture() -> (Majorants, [i32; 6], impl Fn([i32; 3]) -> f32) {
        const BACKGROUND: f32 = 0.125;
        const TILE: f32 = 3.0;
        // The active bounds sit *inside* the stored blocks and off their
        // 8-alignment — the ordinary case, since a leaf is stored whole
        // however few of its voxels are active. It is also what puts a cell
        // boundary strictly inside a block's dilation window on x, which is
        // the only arrangement where the ±1 dilation is load-bearing rather
        // than slack.
        let bbox = [1, 1, 1, 22, 14, 14];
        let mut majorants = Majorants::new(&bbox, BACKGROUND);
        let mut voxels = vec![0.0f32; 16 * 16 * 16];
        let mut state = 0x1234_5678u32;
        let (child_mask, table) = Level::Lower.masks();
        let mut lower = vec![0u8; Level::Lower.stride() as usize];
        for block in 0..8i32 {
            let slot = ((block >> 2) << 8 | (block >> 1 & 1) << 4 | (block & 1)) as usize;
            let origin = [0, 1, 2].map(|axis| 8 * ((block >> (2 - axis)) & 1));
            let mut leaf = vec![0u8; Level::Leaf.stride() as usize];
            for (axis, &at) in origin.iter().enumerate() {
                leaf[4 * axis..4 * axis + 4].copy_from_slice(&at.to_le_bytes());
            }
            for voxel in 0..512usize {
                // Enough spread to cross the floor in both directions, and
                // negative values a bound must not be fooled by.
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let value = (state >> 16) as f32 / 16384.0 - 0.5;
                let at = Level::Leaf.masks().1 + 4 * voxel;
                leaf[at..at + 4].copy_from_slice(&value.to_le_bytes());
                let ijk = [voxel >> 6, voxel >> 3 & 7, voxel & 7];
                let ijk = [0, 1, 2].map(|axis| ijk[axis] + origin[axis] as usize);
                voxels[(ijk[0] * 16 + ijk[1]) * 16 + ijk[2]] = value;
            }
            majorants.scan(&leaf, Level::Leaf, BACKGROUND);
            let word = child_mask + 4 * (slot >> 5);
            let mask = u32::from_le_bytes(lower[word..word + 4].try_into().unwrap());
            lower[word..word + 4].copy_from_slice(&(mask | 1 << (slot & 31)).to_le_bytes());
        }
        // A childless tile at slot (2, 0, 0): voxels [16, 23] × [0, 7]².
        let at = table + 8 * (2 << 8);
        lower[at..at + 4].copy_from_slice(&TILE.to_le_bytes());
        majorants.scan(&lower, Level::Lower, BACKGROUND);
        let field = move |ijk: [i32; 3]| {
            let inside = |lo: i32, hi: i32, at: i32| (lo..=hi).contains(&at);
            if !ijk.iter().all(|&at| inside(0, 127, at)) {
                BACKGROUND // outside the lower node: the root's background
            } else if ijk.iter().all(|&at| inside(0, 15, at)) {
                voxels[((ijk[0] * 16 + ijk[1]) * 16 + ijk[2]) as usize]
            } else if inside(16, 23, ijk[0]) && inside(0, 7, ijk[1]) && inside(0, 7, ijk[2]) {
                TILE
            } else {
                0.0 // a childless tile the fixture left at zero
            }
        };
        (majorants, bbox, field)
    }

    /// The bound, brute-forced. Every value trilinear interpolation can
    /// return anywhere in the shell must sit under the cell the *shader*
    /// would read it against — the cell found by `index · scale + bias`,
    /// floored and clamped exactly as `readCell` does, with `scale` and
    /// `bias` spelled the way `lower_volume` spells them. So this pins the
    /// dilation, the two independent spellings of the lattice transform, and
    /// the node/tile/background coverage in one predicate.
    #[test]
    fn no_interpolated_density_escapes_its_cell() {
        let (majorants, bbox, field) = tree_fixture();
        let res = majorant_resolution(&bbox);
        assert_eq!(res, [3, 2, 2], "the fixture's shell at one cell per 8 voxels");
        assert!(
            majorants.cells.iter().all(|cell| cell.is_finite() && *cell >= 0.0),
            "a ceiling must be a finite non-negative density"
        );
        // `lower_volume`'s arithmetic, not `Majorants`': the record carries
        // these, and a mismatch between them is invisible to every other test.
        let (lo, hi) = shell_box(&bbox);
        let scale: [f32; 3] = std::array::from_fn(|a| res[a] as f32 / (hi[a] - lo[a]));
        let bias: [f32; 3] = std::array::from_fn(|a| -lo[a] * scale[a]);

        let step = 0.25f32;
        let steps: [i32; 3] = std::array::from_fn(|a| ((hi[a] - lo[a]) / step) as i32);
        for z in 0..=steps[2] {
            for y in 0..=steps[1] {
                for x in 0..=steps[0] {
                    let p: [f32; 3] =
                        std::array::from_fn(|a| lo[a] + [x, y, z][a] as f32 * step);
                    let base = p.map(f32::floor);
                    let t: [f32; 3] = std::array::from_fn(|a| p[a] - base[a]);
                    let mut value = 0.0;
                    for corner in 0..8i32 {
                        let bit = |a: usize| (corner >> a) & 1;
                        let ijk = std::array::from_fn(|a| base[a] as i32 + bit(a));
                        let weight: f32 = (0..3)
                            .map(|a| if bit(a) != 0 { t[a] } else { 1.0 - t[a] })
                            .product();
                        value += weight * field(ijk);
                    }
                    let cell: [u32; 3] = std::array::from_fn(|a| {
                        (p[a] * scale[a] + bias[a]).floor().clamp(0.0, (res[a] - 1) as f32)
                            as u32
                    });
                    let ceiling =
                        majorants.cells[((cell[2] * res[1] + cell[1]) * res[0] + cell[0]) as usize];
                    assert!(
                        value <= ceiling,
                        "interpolated {value} at {p:?} escapes cell {cell:?}'s {ceiling}"
                    );
                }
            }
        }
    }

    /// A grid that lies. `NaN` and `+∞` voxels must not reach the lattice:
    /// an infinite ceiling makes the tracker's majorant infinite, where
    /// dropping it leaves the shader's density bound reading those voxels as
    /// whatever finite ceiling surrounds them.
    #[test]
    fn malformed_densities_never_become_ceilings() {
        let bbox = [0, 0, 0, 7, 7, 7];
        let mut majorants = Majorants::new(&bbox, 0.5);
        let table = Level::Leaf.masks().1;
        let mut leaf = vec![0u8; Level::Leaf.stride() as usize];
        for (voxel, value) in [(0usize, f32::NAN), (1, f32::INFINITY), (2, -f32::INFINITY)] {
            leaf[table + 4 * voxel..table + 4 * voxel + 4]
                .copy_from_slice(&value.to_le_bytes());
        }
        majorants.scan(&leaf, Level::Leaf, 0.5);
        assert!(majorants.cells.iter().all(|cell| cell.is_finite()));
        // A finite voxel beside them still lands.
        leaf[table + 12..table + 16].copy_from_slice(&2.0f32.to_le_bytes());
        majorants.scan(&leaf, Level::Leaf, 0.5);
        assert!(majorants.cells.iter().all(|cell| cell.is_finite()));
        assert!(majorants.cells.iter().any(|&cell| cell > 1.9));
    }

    /// The resolution rule: proportional to the shell, never finer than a
    /// leaf, and exactly 1³ at the target the scalar majorant corresponds to.
    #[test]
    fn resolution_tracks_the_shell() {
        assert_eq!(majorant_resolution(&[0, 0, 0, 63, 31, 15]), [9, 5, 3]);
        // A shell smaller than one cell still gets a cell.
        assert_eq!(majorant_resolution(&[0, 0, 0, 0, 0, 0]), [1, 1, 1]);
    }

    #[test]
    fn grid_header_rejects_type_drift() {
        // Container says Float but the payload's own type word disagrees.
        let mut payload = grid_payload(1.0, [0.0; 3], 1.0);
        payload[GRID_OFF_GRID_TYPE..GRID_OFF_GRID_TYPE + 4].copy_from_slice(&2u32.to_le_bytes());
        let bytes = synthesize(&[("density", GRID_TYPE_FLOAT, &payload)]);
        let path = temp_nvdb("drift.nvdb", &bytes);
        let result = grid_header(&path, "density");
        let _ = fs::remove_file(&path);
        assert!(matches!(result, Err(Error::Scene(_))));
    }

    #[test]
    fn bad_magic_is_rejected() {
        let path = temp_nvdb("badmagic.nvdb", &[0u8; 16]);
        let result = read_metadata(&path);
        let _ = fs::remove_file(&path);
        assert!(matches!(result, Err(Error::Scene(_))));
    }

    #[test]
    fn nvdb_passes_through_prepared() {
        let path = Path::new("volumes/cloud.nvdb");
        assert_eq!(prepared(path).expect("pass-through"), path);
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "the lattice is uploaded verbatim; the readback must match bit for bit"
    )]
    fn pool_uploads_dedups_and_grows() {
        let Some(gpu) = crate::gpu::test_context() else {
            return;
        };
        // Empty trees at two backgrounds: the majorant lattice is one cell
        // holding exactly that, which makes the readback below unambiguous.
        let payload_a = grid_payload(1.0, [0.0; 3], 0.5);
        let payload_b = grid_payload(1.0, [0.0; 3], 1.5);
        let path_a = temp_nvdb("pool-a.nvdb", &synthesize(&[("density", 1, &payload_a)]));
        let path_b = temp_nvdb("pool-b.nvdb", &synthesize(&[("density", 1, &payload_b)]));

        let mut pool = GridPool::new();
        let first = pool.upload(&gpu, &path_a, "density").expect("upload a");
        assert_eq!(first.grid, 0);
        // The lattice follows the payload at the next 32-byte boundary.
        assert_eq!(u64::from(first.majorant), 800);
        // Same file, same grid: the existing slots, no growth.
        let again = pool.upload(&gpu, &path_a, "density").expect("re-upload a");
        assert_eq!((again.grid, again.majorant), (first.grid, first.majorant));
        // A second grid grows the pool and lands aligned past the first's
        // four lattice bytes, so the dedup above advanced nothing.
        let second = pool.upload(&gpu, &path_b, "density").expect("upload b");
        assert_eq!(u64::from(second.grid), 832);

        // Both payloads and both lattices survive the growth copy intact.
        let bytes = gpu
            .download_buffer(pool.buffer().expect("resident"))
            .expect("download");
        let cell = |at: u32| {
            f32::from_le_bytes(bytes[at as usize..at as usize + 4].try_into().unwrap())
        };
        assert_eq!(&bytes[..800], &payload_a[..]);
        assert_eq!(cell(first.majorant), 0.5);
        assert_eq!(&bytes[832..832 + 800], &payload_b[..]);
        assert_eq!(cell(second.majorant), 1.5);

        let _ = fs::remove_file(&path_a);
        let _ = fs::remove_file(&path_b);
    }
}
