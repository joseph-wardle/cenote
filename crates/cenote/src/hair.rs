//! Reader for Cem Yuksel's `.hair` files — the format every hair-rendering
//! paper of the last fifteen years benchmarks against, and the only one
//! that ships real grooms (tens of thousands of strands, millions of
//! points) small enough to keep beside a repository.
//!
//! The format is a 128-byte header and then up to five arrays, each
//! present or absent by a flag bit, in a fixed order:
//!
//! ```text
//! "HAIR"  strands  points  flags  default_segments
//! default_thickness  default_transparency  default_color[3]  info[88]
//! segments[strands] u16 | points[3·points] f32 | thickness[points] f32
//! transparency[points] f32 | color[3·points] f32
//! ```
//!
//! Strands are polylines — a strand with `s` segments carries `s + 1`
//! points — so the file *is* the canonical [`Strands`] form, and reading
//! it needs no basis, no flattening, and no faith in a curve type.
//!
//! One convention has to be chosen rather than read: the format calls the
//! per-point value "thickness" without saying whether it is a radius or a
//! diameter. It is taken here as the **width**, which is what pbrt's own
//! `cyhair2pbrt` writes into `width0`/`width1` — so a groom rendered here
//! and in pbrt is the same groom, which is the point of having an oracle.
//! Transparency and color parse as bytes to skip and nothing more; a
//! groom's look belongs to its material.

use std::path::Path;

use glam::Vec3;

use crate::error::{Error, Result};
use crate::scene::curves::Strands;

/// Bit flags naming which arrays follow the header.
mod arrays {
    pub const SEGMENTS: u32 = 1;
    pub const POINTS: u32 = 2;
    pub const THICKNESS: u32 = 4;
    pub const TRANSPARENCY: u32 = 8;
    pub const COLOR: u32 = 16;
}

/// The header's fixed size, and the offset every array is measured from.
const HEADER: usize = 128;

/// Read a groom from a `.hair` file.
///
/// # Errors
///
/// [`Error::Scene`] — a bad groom file is scene data, not a device fault —
/// naming the file and what was wrong with it.
pub(crate) fn read(path: &Path) -> Result<Strands> {
    let bytes = std::fs::read(path)
        .map_err(|error| Error::Scene(format!("hair \"{}\": {error}", path.display())))?;
    parse(&bytes).map_err(|error| match error {
        Error::Scene(message) => Error::Scene(format!("hair \"{}\": {message}", path.display())),
        other => other,
    })
}

/// Parse a whole file into strands.
fn parse(bytes: &[u8]) -> Result<Strands> {
    if bytes.len() < HEADER || &bytes[..4] != b"HAIR" {
        return Err(scene(
            "not a HAIR file (the first four bytes are its signature)",
        ));
    }
    let strand_count = u32(bytes, 4) as usize;
    let point_count = u32(bytes, 8) as usize;
    let flags = u32(bytes, 12);
    let default_segments = u32(bytes, 16) as usize;
    let default_width = f32(bytes, 20);
    if flags & arrays::POINTS == 0 {
        return Err(scene("carries no points array"));
    }

    // Every array's extent, in the order the format writes them.
    let segments_at = HEADER;
    let points_at = segments_at + if flags & arrays::SEGMENTS != 0 { 2 * strand_count } else { 0 };
    let thickness_at = points_at + 12 * point_count;
    let end = thickness_at
        + if flags & arrays::THICKNESS != 0 { 4 * point_count } else { 0 }
        + if flags & arrays::TRANSPARENCY != 0 { 4 * point_count } else { 0 }
        + if flags & arrays::COLOR != 0 { 12 * point_count } else { 0 };
    if bytes.len() < end {
        return Err(scene(format!(
            "is {} bytes, but its header describes {end}",
            bytes.len()
        )));
    }

    let counts: Vec<usize> = if flags & arrays::SEGMENTS != 0 {
        (0..strand_count)
            .map(|strand| {
                let at = segments_at + 2 * strand;
                usize::from(u16::from_le_bytes([bytes[at], bytes[at + 1]])) + 1
            })
            .collect()
    } else {
        vec![default_segments + 1; strand_count]
    };
    let described: usize = counts.iter().sum();
    if described != point_count {
        return Err(scene(format!(
            "describes {described} points across its strands, but its header declares \
             {point_count}"
        )));
    }

    let mut strands = Strands::new();
    strands.reserve(strand_count, point_count);
    let mut point = 0usize;
    for count in counts {
        for index in point..point + count {
            let at = points_at + 12 * index;
            let position = Vec3::new(f32(bytes, at), f32(bytes, at + 4), f32(bytes, at + 8));
            let width = if flags & arrays::THICKNESS != 0 {
                f32(bytes, thickness_at + 4 * index)
            } else {
                default_width
            };
            strands.push_point(position, 0.5 * width);
        }
        strands.end_strand();
        point += count;
    }
    Ok(strands)
}

fn scene(message: impl Into<String>) -> Error {
    Error::Scene(message.into())
}

/// A little-endian `u32` at `offset` — every scalar in the format is one
/// of these or an `f32` beside it.
fn u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn f32(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal file: two strands of three points, no optional arrays
    /// beyond the required positions.
    fn fixture(flags: u32, extra: &[u8], segments: Option<&[u16]>, points: &[[f32; 3]]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"HAIR");
        bytes.extend_from_slice(&(segments.map_or(2u32, |s| s.len() as u32)).to_le_bytes());
        bytes.extend_from_slice(&(points.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&flags.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes()); // default segments
        bytes.extend_from_slice(&0.2f32.to_le_bytes()); // default thickness
        bytes.extend_from_slice(&1.0f32.to_le_bytes()); // default transparency
        bytes.extend_from_slice(&[0; 12]); // default color
        bytes.extend_from_slice(&[0; 88]); // info
        assert_eq!(bytes.len(), HEADER);
        if let Some(segments) = segments {
            for count in segments {
                bytes.extend_from_slice(&count.to_le_bytes());
            }
        }
        for point in points {
            for axis in point {
                bytes.extend_from_slice(&axis.to_le_bytes());
            }
        }
        bytes.extend_from_slice(extra);
        bytes
    }

    fn line(count: usize, offset: f32) -> Vec<[f32; 3]> {
        (0..count)
            .map(|index| [offset, index as f32, 0.0])
            .collect()
    }

    #[test]
    fn default_segments_and_thickness_apply_when_the_arrays_are_absent() {
        let mut points = line(3, 0.0);
        points.extend(line(3, 1.0));
        let bytes = fixture(arrays::POINTS, &[], None, &points);
        let strands = parse(&bytes).expect("a valid groom");
        assert_eq!(strands.len(), 2);
        assert_eq!(strands.points(), 6);
        // The header's thickness is a width, so the radius is half of it.
        assert!((strands.radius(0) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn per_strand_segment_counts_partition_the_points() {
        let mut points = line(2, 0.0);
        points.extend(line(4, 1.0));
        let bytes = fixture(arrays::POINTS | arrays::SEGMENTS, &[], Some(&[1, 3]), &points);
        let strands = parse(&bytes).expect("a valid groom");
        assert_eq!(strands.len(), 2);
        assert_eq!(strands.points(), 6);
    }

    #[test]
    fn per_point_thickness_wins_over_the_header() {
        let points = line(3, 0.0);
        let mut widths = Vec::new();
        for width in [1.0f32, 0.5, 0.0] {
            widths.extend_from_slice(&width.to_le_bytes());
        }
        let bytes = fixture(
            arrays::POINTS | arrays::THICKNESS | arrays::SEGMENTS,
            &widths,
            Some(&[2]),
            &points,
        );
        let strands = parse(&bytes).expect("a valid groom");
        assert!((strands.radius(0) - 0.5).abs() < 1e-6);
        assert!((strands.radius(2) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn a_truncated_file_is_a_scene_error() {
        let points = line(3, 0.0);
        let mut bytes = fixture(arrays::POINTS | arrays::COLOR, &[], Some(&[2]), &points);
        bytes.truncate(bytes.len() - 4);
        let message = parse(&bytes).expect_err("the color array is missing");
        assert!(format!("{message}").contains("bytes"), "{message}");
    }

    #[test]
    fn a_header_that_disagrees_with_its_segments_is_a_scene_error() {
        let points = line(3, 0.0);
        let bytes = fixture(arrays::POINTS | arrays::SEGMENTS, &[], Some(&[9]), &points);
        let message = parse(&bytes).expect_err("ten points were promised, three delivered");
        assert!(format!("{message}").contains("declares"), "{message}");
    }

    #[test]
    fn a_file_without_the_signature_is_refused() {
        let mut bytes = fixture(arrays::POINTS, &[], Some(&[2]), &line(3, 0.0));
        bytes[0] = b'X';
        parse(&bytes).expect_err("the signature is the format check");
    }
}
