//! Hand-rolled PLY reader — the bulk-geometry half of the mesh schema.
//! Small meshes live inline in scene files; heavy geometry stays in PLY,
//! the format the pbrt corpus (and everyone else) already ships. The
//! format is small enough to read by hand: a self-describing header names
//! elements and their typed properties, then the data follows in declared
//! order, ASCII or binary.
//!
//! Read here: `vertex` positions (required), normals and UVs when
//! present, and `face` index lists, fan-triangulated so quad-dominant
//! exports work. Everything else — extra properties (vertex colors,
//! confidence), whole unknown elements — parses and drops, because a
//! reader that only *skips bytes it understands* can still walk to the
//! data it wants. ASCII and binary little-endian cover every file in
//! practice; big-endian is refused by name rather than silently
//! byte-swapped wrong.

use std::path::Path;

use glam::{Vec2, Vec3};

use crate::error::{Error, Result};

/// A PLY file's renderable contents, shaped like the inline mesh payload:
/// optional streams stay optional so prep applies the same
/// derive-or-zero-fill rules to both sources.
#[derive(Debug)]
pub(crate) struct Ply {
    /// Vertex positions (`x`, `y`, `z`).
    pub positions: Vec<Vec3>,
    /// Shading normals, present only when the file carries all of `nx`,
    /// `ny`, `nz`.
    pub normals: Option<Vec<Vec3>>,
    /// Texture coordinates, under any of the customary property-name
    /// pairs (`u`/`v`, `s`/`t`, `texture_u`/`texture_v`).
    pub uvs: Option<Vec<Vec2>>,
    /// Triangles, fan-triangulated from the file's polygons, indices
    /// validated in bounds.
    pub triangles: Vec<[u32; 3]>,
}

/// Read a mesh from a PLY file.
///
/// # Errors
///
/// [`Error::Scene`] — a bad mesh file is scene data, not a device fault —
/// naming the file and what was wrong with it.
pub(crate) fn read(path: &Path) -> Result<Ply> {
    let bytes = std::fs::read(path)
        .map_err(|error| Error::Scene(format!("PLY \"{}\": {error}", path.display())))?;
    parse(&bytes).map_err(|error| match error {
        Error::Scene(message) => Error::Scene(format!("PLY \"{}\": {message}", path.display())),
        other => other,
    })
}

/// The scalar types PLY properties are declared with (both spelling
/// generations: `float` and `float32` mean the same thing).
#[derive(Clone, Copy)]
enum Scalar {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl Scalar {
    fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "char" | "int8" => Self::I8,
            "uchar" | "uint8" => Self::U8,
            "short" | "int16" => Self::I16,
            "ushort" | "uint16" => Self::U16,
            "int" | "int32" => Self::I32,
            "uint" | "uint32" => Self::U32,
            "float" | "float32" => Self::F32,
            "double" | "float64" => Self::F64,
            _ => return Err(scene(format!("unknown property type \"{name}\""))),
        })
    }

    fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

/// One declared property: a scalar, or a count-prefixed list.
enum Property {
    Scalar {
        ty: Scalar,
        name: String,
    },
    List {
        count: Scalar,
        item: Scalar,
        name: String,
    },
}

struct Element {
    name: String,
    count: usize,
    properties: Vec<Property>,
}

/// The two data encodings this reader speaks. Every scalar comes out as
/// `f64` — wide enough to hold any of the eight types exactly, including
/// full-range `u32` indices.
enum Data<'a> {
    Ascii(std::str::SplitAsciiWhitespace<'a>),
    Binary(&'a [u8]),
}

impl Data<'_> {
    fn scalar(&mut self, ty: Scalar) -> Result<f64> {
        match self {
            Self::Ascii(tokens) => {
                let token = tokens
                    .next()
                    .ok_or_else(|| scene("data ends before its elements do".into()))?;
                token
                    .parse()
                    .map_err(|_| scene(format!("\"{token}\" is not a number")))
            }
            Self::Binary(bytes) => {
                let (value, rest) = bytes
                    .split_at_checked(ty.size())
                    .ok_or_else(|| scene("data ends before its elements do".into()))?;
                *bytes = rest;
                Ok(match ty {
                    Scalar::I8 => f64::from(value[0].cast_signed()),
                    Scalar::U8 => f64::from(value[0]),
                    Scalar::I16 => f64::from(i16::from_le_bytes([value[0], value[1]])),
                    Scalar::U16 => f64::from(u16::from_le_bytes([value[0], value[1]])),
                    Scalar::I32 => {
                        f64::from(i32::from_le_bytes([value[0], value[1], value[2], value[3]]))
                    }
                    Scalar::U32 => {
                        f64::from(u32::from_le_bytes([value[0], value[1], value[2], value[3]]))
                    }
                    Scalar::F32 => {
                        f64::from(f32::from_le_bytes([value[0], value[1], value[2], value[3]]))
                    }
                    Scalar::F64 => f64::from_le_bytes(value.try_into().expect("split gave 8")),
                })
            }
        }
    }
}

fn scene(message: String) -> Error {
    Error::Scene(message)
}

fn parse(bytes: &[u8]) -> Result<Ply> {
    let (elements, mut data) = parse_header(bytes)?;
    let mut mesh = Ply {
        positions: Vec::new(),
        normals: None,
        uvs: None,
        triangles: Vec::new(),
    };
    // Elements must be walked in declared order — each one's data starts
    // where the previous one's ends — so unknown elements are read and
    // dropped, not jumped over.
    for element in &elements {
        match element.name.as_str() {
            "vertex" => read_vertices(element, &mut data, &mut mesh)?,
            "face" => read_faces(element, &mut data, &mut mesh)?,
            _ => {
                for _ in 0..element.count {
                    for property in &element.properties {
                        read_and_drop(property, &mut data)?;
                    }
                }
            }
        }
    }
    if mesh.positions.is_empty() || mesh.triangles.is_empty() {
        return Err(scene("the file has no geometry".into()));
    }
    Ok(mesh)
}

/// Parse the header lines, returning the declared elements and the data
/// section wrapped in its declared encoding.
fn parse_header(bytes: &[u8]) -> Result<(Vec<Element>, Data<'_>)> {
    let mut cursor = 0;
    let mut lines = Vec::new();
    let data_start = loop {
        let end = bytes[cursor..]
            .iter()
            .position(|&byte| byte == b'\n')
            .ok_or_else(|| scene("header has no end_header line".into()))?
            + cursor;
        let line = std::str::from_utf8(&bytes[cursor..end])
            .map_err(|_| scene("header is not valid UTF-8".into()))?
            .trim();
        cursor = end + 1;
        if line == "end_header" {
            break cursor;
        }
        lines.push(line);
    };

    let mut lines = lines.into_iter();
    if lines.next() != Some("ply") {
        return Err(scene("not a PLY file (missing the \"ply\" magic)".into()));
    }
    let mut binary = None;
    let mut elements: Vec<Element> = Vec::new();
    for line in lines {
        let mut words = line.split_ascii_whitespace();
        match words.next() {
            Some("format") => {
                binary = Some(match words.next() {
                    Some("ascii") => false,
                    Some("binary_little_endian") => true,
                    Some(other) => {
                        return Err(scene(format!("unsupported format \"{other}\"")));
                    }
                    None => return Err(scene("format line names no encoding".into())),
                });
            }
            Some("comment" | "obj_info") | None => {}
            Some("element") => {
                let name = words
                    .next()
                    .ok_or_else(|| scene("element line names no element".into()))?;
                let count = words
                    .next()
                    .and_then(|count| count.parse().ok())
                    .ok_or_else(|| scene(format!("element \"{name}\" has no valid count")))?;
                elements.push(Element {
                    name: name.to_owned(),
                    count,
                    properties: Vec::new(),
                });
            }
            Some("property") => {
                let element = elements
                    .last_mut()
                    .ok_or_else(|| scene("property declared before any element".into()))?;
                element.properties.push(parse_property(&mut words)?);
            }
            Some(other) => {
                return Err(scene(format!("unknown header line \"{other}\"")));
            }
        }
    }
    let Some(binary) = binary else {
        return Err(scene("header has no format line".into()));
    };
    let data = if binary {
        Data::Binary(&bytes[data_start..])
    } else {
        Data::Ascii(
            std::str::from_utf8(&bytes[data_start..])
                .map_err(|_| scene("ASCII data is not valid UTF-8".into()))?
                .split_ascii_whitespace(),
        )
    };
    Ok((elements, data))
}

/// Parse one `property` header line (past the keyword itself).
fn parse_property(words: &mut std::str::SplitAsciiWhitespace<'_>) -> Result<Property> {
    let ty = words
        .next()
        .ok_or_else(|| scene("property line names no type".into()))?;
    if ty == "list" {
        let count = Scalar::parse(
            words
                .next()
                .ok_or_else(|| scene("list property names no count type".into()))?,
        )?;
        let item = Scalar::parse(
            words
                .next()
                .ok_or_else(|| scene("list property names no item type".into()))?,
        )?;
        let name = words
            .next()
            .ok_or_else(|| scene("list property has no name".into()))?;
        Ok(Property::List {
            count,
            item,
            name: name.to_owned(),
        })
    } else {
        let ty = Scalar::parse(ty)?;
        let name = words
            .next()
            .ok_or_else(|| scene("property has no name".into()))?;
        Ok(Property::Scalar {
            ty,
            name: name.to_owned(),
        })
    }
}

/// Read the vertex element: positions required, normals and UVs when
/// their customary property names are all present, anything else dropped.
fn read_vertices(element: &Element, data: &mut Data<'_>, mesh: &mut Ply) -> Result<()> {
    let index_of = |wanted: &str| {
        element.properties.iter().position(
            |property| matches!(property, Property::Scalar { name, .. } if name == wanted),
        )
    };
    let triple = |names: [&str; 3]| {
        Some([
            index_of(names[0])?,
            index_of(names[1])?,
            index_of(names[2])?,
        ])
    };
    let position = triple(["x", "y", "z"])
        .ok_or_else(|| scene("vertex element lacks x/y/z properties".into()))?;
    let normal = triple(["nx", "ny", "nz"]);
    let uv = [["u", "v"], ["s", "t"], ["texture_u", "texture_v"]]
        .iter()
        .find_map(|names| Some([index_of(names[0])?, index_of(names[1])?]));

    // The header count is untrusted: cap the pre-reservation so a corrupt
    // "element vertex 999999999999" can't demand a terabyte before a single
    // byte is read. The read loop is already bounded — it hits EOF and
    // errors out — so an honest mesh larger than the cap just grows.
    let reserve = element.count.min(1 << 20);
    mesh.positions.reserve(reserve);
    let mut normals = normal.map(|_| Vec::with_capacity(reserve));
    let mut uvs = uv.map(|_| Vec::with_capacity(reserve));
    let mut row = vec![0.0f64; element.properties.len()];
    for _ in 0..element.count {
        for (slot, property) in row.iter_mut().zip(&element.properties) {
            match property {
                Property::Scalar { ty, .. } => *slot = data.scalar(*ty)?,
                // A list on a vertex (some scanners emit one) has no fixed
                // slot; read it out of the way.
                list @ Property::List { .. } => read_and_drop(list, data)?,
            }
        }
        let at = |index: usize| row[index] as f32;
        mesh.positions
            .push(Vec3::new(at(position[0]), at(position[1]), at(position[2])));
        if let (Some(normals), Some([nx, ny, nz])) = (normals.as_mut(), normal) {
            normals.push(Vec3::new(at(nx), at(ny), at(nz)));
        }
        if let (Some(uvs), Some([u, v])) = (uvs.as_mut(), uv) {
            uvs.push(Vec2::new(at(u), at(v)));
        }
    }
    mesh.normals = normals;
    mesh.uvs = uvs;
    Ok(())
}

/// Read the face element, fan-triangulating each polygon around its first
/// corner and bounds-checking every index — an index off the end would
/// otherwise ride silently into GPU buffers.
fn read_faces(element: &Element, data: &mut Data<'_>, mesh: &mut Ply) -> Result<()> {
    let vertex_count = mesh.positions.len() as f64;
    for _ in 0..element.count {
        for property in &element.properties {
            let Property::List { count, item, name } = property else {
                read_and_drop(property, data)?;
                continue;
            };
            if !matches!(name.as_str(), "vertex_indices" | "vertex_index") {
                read_and_drop(property, data)?;
                continue;
            }
            let corners = data.scalar(*count)? as usize;
            if corners < 3 {
                return Err(scene(format!("a face has {corners} corners")));
            }
            let index = |data: &mut Data<'_>| -> Result<u32> {
                let value = data.scalar(*item)?;
                if value < 0.0 || value >= vertex_count {
                    return Err(scene(format!(
                        "face index {value} is out of bounds for {} vertices",
                        mesh.positions.len()
                    )));
                }
                Ok(value as u32)
            };
            let first = index(data)?;
            let mut previous = index(data)?;
            for _ in 2..corners {
                let corner = index(data)?;
                mesh.triangles.push([first, previous, corner]);
                previous = corner;
            }
        }
    }
    Ok(())
}

/// Consume one property's data without keeping it — how unknown
/// properties and whole unknown elements stay skippable in both encodings.
fn read_and_drop(property: &Property, data: &mut Data<'_>) -> Result<()> {
    match property {
        Property::Scalar { ty, .. } => {
            data.scalar(*ty)?;
        }
        Property::List { count, item, .. } => {
            let length = data.scalar(*count)? as usize;
            for _ in 0..length {
                data.scalar(*item)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A featureful ASCII file: comments, double-typed positions, extra
    /// vertex properties to skip, normals, `s`/`t` UVs, and a quad that
    /// must fan into two triangles.
    #[test]
    fn ascii_reads_streams_and_fan_triangulates() {
        let text = "\
ply
comment a unit quad from a scanner with opinions
format ascii 1.0
element vertex 4
property double x
property double y
property double z
property float nx
property float ny
property float nz
property uchar red
property float s
property float t
element face 1
property list uchar int vertex_indices
end_header
0 0 0  0 1 0  255  0 0
1 0 0  0 1 0  255  1 0
1 0 1  0 1 0  255  1 1
0 0 1  0 1 0  255  0 1
4 0 1 2 3
";
        let mesh = parse(text.as_bytes()).expect("parses");
        assert_eq!(mesh.positions.len(), 4);
        assert_eq!(mesh.positions[2], Vec3::new(1.0, 0.0, 1.0));
        assert_eq!(mesh.normals.as_deref(), Some(&[Vec3::Y; 4][..]));
        let uvs = mesh.uvs.expect("s/t read as UVs");
        assert_eq!(uvs[2], Vec2::new(1.0, 1.0));
        assert_eq!(mesh.triangles, vec![[0, 1, 2], [0, 2, 3]]);
    }

    /// The same quad in binary little-endian, with a `ushort` index type
    /// and a `uint` list count for coverage of the width table.
    #[test]
    fn binary_little_endian_reads() {
        let mut bytes = b"ply\r\nformat binary_little_endian 1.0\r\n\
             element vertex 3\r\nproperty float x\r\nproperty float y\r\nproperty float z\r\n\
             element face 1\r\nproperty list uint ushort vertex_indices\r\nend_header\r\n"
            .to_vec();
        for position in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
            for value in position {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
        }
        bytes.extend_from_slice(&3u32.to_le_bytes());
        for index in [0u16, 1, 2] {
            bytes.extend_from_slice(&index.to_le_bytes());
        }
        let mesh = parse(&bytes).expect("parses");
        assert_eq!(mesh.positions[1], Vec3::X);
        assert!(mesh.normals.is_none());
        assert!(mesh.uvs.is_none());
        assert_eq!(mesh.triangles, vec![[0, 1, 2]]);
    }

    /// An element this reader has never heard of sits between vertex and
    /// face; its rows (including a list) must be consumed, not jumped
    /// over, for the face data to line up.
    #[test]
    fn unknown_elements_are_read_past() {
        let text = "\
ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
element edge 2
property int vertex1
property int vertex2
property list uchar float weights
element face 1
property list uchar int vertex_index
end_header
0 0 0
1 0 0
0 1 0
0 1 2 0.5 0.5
1 2 3 0.1 0.2 0.3
3 0 1 2
";
        let mesh = parse(text.as_bytes()).expect("parses");
        assert_eq!(mesh.triangles, vec![[0, 1, 2]]);
    }

    #[test]
    fn malformed_files_are_refused_by_name() {
        let refused = |text: &str, needle: &str| {
            let error = parse(text.as_bytes()).unwrap_err();
            assert!(error.to_string().contains(needle), "{error}");
        };
        refused("not a ply\n", "end_header");
        refused("obj\nformat ascii 1.0\nend_header\n", "magic");
        refused(
            "ply\nformat binary_big_endian 1.0\nend_header\n",
            "binary_big_endian",
        );
        refused(
            "ply\nformat ascii 1.0\nelement vertex 1\nproperty float w\n\
             element face 0\nend_header\n1.0\n",
            "x/y/z",
        );
        // Out-of-bounds index.
        refused(
            "ply\nformat ascii 1.0\nelement vertex 3\n\
             property float x\nproperty float y\nproperty float z\n\
             element face 1\nproperty list uchar int vertex_indices\nend_header\n\
             0 0 0 1 0 0 0 1 0\n3 0 1 9\n",
            "out of bounds",
        );
        // Truncated data.
        refused(
            "ply\nformat ascii 1.0\nelement vertex 2\n\
             property float x\nproperty float y\nproperty float z\nend_header\n0 0 0 1\n",
            "ends before",
        );
        refused(
            "ply\nformat ascii 1.0\nelement vertex 1\n\
             property float x\nproperty float y\nproperty float z\nend_header\n0 zero 0\n",
            "not a number",
        );
    }

    #[test]
    fn a_missing_file_names_its_path() {
        let error = read(Path::new("/no/such/mesh.ply")).unwrap_err();
        assert!(error.to_string().contains("/no/such/mesh.ply"), "{error}");
    }

    /// A wildly overstated vertex count must fail on the missing data, not
    /// try to pre-allocate a terabyte from the untrusted header.
    #[test]
    fn an_absurd_vertex_count_errors_before_it_allocates() {
        let text = "\
ply
format ascii 1.0
element vertex 999999999999
property float x
property float y
property float z
end_header
0 0 0
";
        let error = parse(text.as_bytes()).unwrap_err();
        assert!(matches!(error, Error::Scene(_)), "{error}");
    }
}
