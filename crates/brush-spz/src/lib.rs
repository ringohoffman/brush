use brush_render::gaussian_splats::Splats;
use burn::prelude::Backend;
use burn::tensor::Transaction;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::io::{Cursor, Read, Write};

const MAGIC: u32 = 0x5053474e; // "NGSP"
const VERSION: u32 = 3;

/// Scale factor for DC color components. To convert to RGB, we should multiply by 0.282, but it can
/// be useful to represent base colors that are out of range if the higher spherical harmonics bands
/// bring them back into range so we multiply by a smaller value.
const COLOR_SCALE: f32 = 0.15;

fn to_uint8(x: f32) -> u8 {
    x.round().clamp(0.0, 255.0) as u8
}

/// Quantizes SH coefficients to 8 bits, rounding to the nearest bucket center.
/// The bucket_size determines the quantization granularity (e.g., 8 for 5-bit, 16 for 4-bit).
fn quantize_sh(x: f32, bucket_size: i32) -> u8 {
    let q = (x * 128.0).round() as i32 + 128;
    let q = (q + bucket_size / 2) / bucket_size * bucket_size;
    q.clamp(0, 255) as u8
}

/// Unquantizes an 8-bit SH coefficient back to float.
fn unquantize_sh(x: u8) -> f32 {
    (x as f32 - 128.0) / 128.0
}

/// Packs a quaternion using "smallest three" compression.
/// Input quaternion q_in is in [x, y, z, w] order (matching C++ SPZ internal format).
/// This matches the C++ reference implementation exactly.
fn pack_quaternion_smallest_three(r: &mut [u8], q_in: [f32; 4]) {
    // Normalize the quaternion
    let norm =
        (q_in[0] * q_in[0] + q_in[1] * q_in[1] + q_in[2] * q_in[2] + q_in[3] * q_in[3]).sqrt();
    let q = [
        q_in[0] / norm,
        q_in[1] / norm,
        q_in[2] / norm,
        q_in[3] / norm,
    ];

    // Find the component with largest absolute value
    let mut i_largest = 0usize;
    for i in 1..4 {
        if q[i].abs() > q[i_largest].abs() {
            i_largest = i;
        }
    }

    // Since -q represents the same rotation as q, ensure largest component is positive
    // to avoid sending its sign bit
    let negate = q[i_largest] < 0.0;

    // Pack using sign bit and 9-bit precision per element
    // C++ iterates 0, 1, 2, 3 and stores iLargest directly in top 2 bits
    let mut comp = i_largest as u32;
    let sqrt1_2: f32 = 0.7071067811865476;

    for i in 0..4 {
        if i != i_largest {
            let negbit = (q[i] < 0.0) ^ negate;
            // Match C++ reference expression exactly: (uint32_t)(float((1u << 9u) - 1u) * (std::fabs(q[i]) / sqrt1_2) + 0.5f)
            let mag = (511.0f32 * (q[i].abs() / sqrt1_2) + 0.5f32) as u32;
            let mag = mag.min(511);
            comp = (comp << 10) | ((if negbit { 1 } else { 0 }) << 9) | mag;
        }
    }

    // Ensure little-endianness
    r[0] = (comp & 0xff) as u8;
    r[1] = ((comp >> 8) & 0xff) as u8;
    r[2] = ((comp >> 16) & 0xff) as u8;
    r[3] = ((comp >> 24) & 0xff) as u8;
}

/// Unpacks a quaternion from the "smallest three" representation.
/// Returns quaternion in [x, y, z, w] order (matching C++ SPZ internal format).
/// This matches the C++ reference implementation exactly.
fn unpack_quaternion_smallest_three(r: &[u8]) -> [f32; 4] {
    let mut comp = r[0] as u32 | (r[1] as u32) << 8 | (r[2] as u32) << 16 | (r[3] as u32) << 24;

    const C_MASK: u32 = (1u32 << 9) - 1;
    let sqrt1_2: f32 = 0.7071067811865476;

    // i_largest is stored directly in top 2 bits (no offset)
    let i_largest = (comp >> 30) as usize;

    let mut rotation = [0.0f32; 4];
    let mut sum_squares = 0.0f32;

    // Unpack in reverse order (3, 2, 1, 0) matching the C++ reference
    for i in (0..4).rev() {
        if i != i_largest {
            let mag = comp & C_MASK;
            let negbit = (comp >> 9) & 0x1;
            comp >>= 10;
            rotation[i] = sqrt1_2 * (mag as f32) / (C_MASK as f32);
            if negbit == 1 {
                rotation[i] = -rotation[i];
            }
            sum_squares += rotation[i] * rotation[i];
        }
    }
    rotation[i_largest] = (1.0 - sum_squares).max(0.0).sqrt();

    rotation
}

/// Returns the number of REST SH coefficients (excluding DC) for a given SH degree.
/// This matches the SPZ format's dimForDegree function.
fn sh_rest_dim_for_degree(degree: u32) -> usize {
    match degree {
        0 => 0,
        1 => 3,
        2 => 8,
        3 => 15,
        _ => 0,
    }
}

pub async fn splat_to_spz<B: Backend>(splats: Splats<B>) -> anyhow::Result<Vec<u8>> {
    let sh_degree = splats.sh_degree();
    let num_points = splats.num_splats() as usize;

    let [means, log_scales, rotations, raw_opacities, sh_coeffs] = Transaction::default()
        .register(splats.means.val())
        .register(splats.log_scales.val())
        .register(splats.rotations.val())
        .register(splats.raw_opacities.val())
        .register(splats.sh_coeffs.val())
        .execute_async()
        .await
        .expect("Failed to fetch splat data")
        .into_iter()
        .map(|x| x.into_vec::<f32>().unwrap())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    // sh_coeffs is [N, coeffs_per_channel, 3] flattened
    // where coeffs_per_channel = (sh_degree + 1)^2
    let coeffs_per_channel = (sh_degree + 1).pow(2) as usize;

    // SPZ stores only the REST coefficients (excluding DC) in the sh array
    let sh_rest_dim = sh_rest_dim_for_degree(sh_degree);

    let mut packed_positions = Vec::with_capacity(num_points * 9);
    let mut packed_scales = Vec::with_capacity(num_points * 3);
    let mut packed_rotations = Vec::with_capacity(num_points * 4);
    let mut packed_alphas = Vec::with_capacity(num_points);
    let mut packed_colors = Vec::with_capacity(num_points * 3);
    let mut packed_sh = Vec::with_capacity(num_points * sh_rest_dim * 3);

    let fractional_bits = 12;
    let scale_pos = (1 << fractional_bits) as f32;

    // Quantization parameters for SH coefficients (matching reference implementation)
    // First 3 rest coefficients (degree 1) use 5-bit precision -> bucket_size = 8
    // Remaining coefficients use 4-bit precision -> bucket_size = 16
    let sh1_bucket_size = 1 << (8 - 5); // 8
    let sh_rest_bucket_size = 1 << (8 - 4); // 16

    for i in 0..num_points {
        // Position: 24-bit fixed point
        for j in 0..3 {
            let pos = means[i * 3 + j];
            let fixed32 = (pos * scale_pos).round() as i32;
            packed_positions.push((fixed32 & 0xff) as u8);
            packed_positions.push(((fixed32 >> 8) & 0xff) as u8);
            packed_positions.push(((fixed32 >> 16) & 0xff) as u8);
        }

        // Scales: uint8
        for j in 0..3 {
            let s = log_scales[i * 3 + j];
            packed_scales.push(to_uint8((s + 10.0) * 16.0));
        }

        // Rotations: smallest three
        // Brush stores rotations as [w, x, y, z] at indices [0, 1, 2, 3]
        // SPZ C++ expects [x, y, z, w] order, so convert
        let mut r_bytes = [0u8; 4];
        let q = [
            rotations[i * 4 + 1], // x
            rotations[i * 4 + 2], // y
            rotations[i * 4 + 3], // z
            rotations[i * 4 + 0], // w
        ];
        pack_quaternion_smallest_three(&mut r_bytes, q);
        packed_rotations.extend_from_slice(&r_bytes);

        // Alpha: uint8 (sigmoid)
        let raw_opac = raw_opacities[i];
        let alpha = 1.0 / (1.0 + (-raw_opac).exp());
        packed_alphas.push(to_uint8(alpha * 255.0));

        // Colors (DC): uint8
        // sh_coeffs layout: [N, coeffs_per_channel, 3] flattened
        // For point i, DC coefficient (index 0) for each channel:
        // - Red DC: sh_coeffs[i * coeffs_per_channel * 3 + 0 * 3 + 0]
        // - Green DC: sh_coeffs[i * coeffs_per_channel * 3 + 0 * 3 + 1]
        // - Blue DC: sh_coeffs[i * coeffs_per_channel * 3 + 0 * 3 + 2]
        let sh_base = i * coeffs_per_channel * 3;
        for j in 0..3 {
            let dc = sh_coeffs[sh_base + j];
            packed_colors.push(to_uint8(dc * (COLOR_SCALE * 255.0) + (0.5 * 255.0)));
        }

        // SH rest coefficients: stored as [coeff_index, channel] interleaved per point
        // SPZ expects: for each point, iterate over rest coeff indices (1..coeffs_per_channel),
        // and for each, store R, G, B
        if sh_degree > 0 {
            for k in 1..coeffs_per_channel {
                // k is the coefficient index (1 to coeffs_per_channel-1 for rest)
                for j in 0..3 {
                    // j is the channel (R, G, B)
                    let val = sh_coeffs[sh_base + k * 3 + j];

                    // Apply quantization: first 3 rest coeffs (k=1,2,3) use 5-bit, rest use 4-bit
                    let bucket_size = if k <= 3 {
                        sh1_bucket_size
                    } else {
                        sh_rest_bucket_size
                    };
                    packed_sh.push(quantize_sh(val, bucket_size));
                }
            }
        }
    }

    let mut uncompressed = Vec::new();
    uncompressed.write_u32::<LittleEndian>(MAGIC)?;
    uncompressed.write_u32::<LittleEndian>(VERSION)?;
    uncompressed.write_u32::<LittleEndian>(num_points as u32)?;
    uncompressed.write_u8(sh_degree as u8)?;
    uncompressed.write_u8(fractional_bits as u8)?;
    uncompressed.write_u8(0)?; // flags
    uncompressed.write_u8(0)?; // reserved

    uncompressed.extend_from_slice(&packed_positions);
    uncompressed.extend_from_slice(&packed_alphas);
    uncompressed.extend_from_slice(&packed_colors);
    uncompressed.extend_from_slice(&packed_scales);
    uncompressed.extend_from_slice(&packed_rotations);
    uncompressed.extend_from_slice(&packed_sh);

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(&uncompressed)?;
    let compressed = encoder.finish()?;

    Ok(compressed)
}

/// Decoded SPZ data as raw vectors (not yet converted to Splats).
#[derive(Debug, Clone)]
pub struct DecodedSpz {
    pub num_points: usize,
    pub sh_degree: u32,
    /// Positions [N * 3] flattened
    pub positions: Vec<f32>,
    /// Log scales [N * 3] flattened
    pub log_scales: Vec<f32>,
    /// Rotations [N * 4] in [x, y, z, w] order, flattened
    pub rotations: Vec<f32>,
    /// Raw opacities [N] (inverse sigmoid of alpha)
    pub raw_opacities: Vec<f32>,
    /// SH coefficients [N, coeffs_per_channel, 3] flattened
    pub sh_coeffs: Vec<f32>,
}

/// Inverse sigmoid function: log(x / (1 - x))
fn inv_sigmoid(x: f32) -> f32 {
    // Clamp to avoid infinity at boundaries
    let x = x.clamp(1e-6, 1.0 - 1e-6);
    (x / (1.0 - x)).ln()
}

/// Decodes an SPZ file into raw vectors.
pub fn spz_to_raw(data: &[u8]) -> anyhow::Result<DecodedSpz> {
    let mut decoder = GzDecoder::new(data);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)?;

    let mut cursor = Cursor::new(&decompressed);

    // Read header
    let magic = cursor.read_u32::<LittleEndian>()?;
    if magic != MAGIC {
        anyhow::bail!("Invalid SPZ magic: expected 0x{:08X}, got 0x{:08X}", MAGIC, magic);
    }

    let version = cursor.read_u32::<LittleEndian>()?;
    if version < 1 || version > 3 {
        anyhow::bail!("Unsupported SPZ version: {}", version);
    }

    let num_points = cursor.read_u32::<LittleEndian>()? as usize;
    let sh_degree = cursor.read_u8()? as u32;
    let fractional_bits = cursor.read_u8()?;
    let _flags = cursor.read_u8()?;
    let _reserved = cursor.read_u8()?;

    if sh_degree > 3 {
        anyhow::bail!("Unsupported SH degree: {}", sh_degree);
    }

    let sh_rest_dim = sh_rest_dim_for_degree(sh_degree);
    let coeffs_per_channel = (sh_degree + 1).pow(2) as usize;

    // Calculate expected data sizes
    let uses_quaternion_smallest_three = version >= 3;
    let position_bytes = num_points * 9; // 3 components * 3 bytes each
    let alpha_bytes = num_points;
    let color_bytes = num_points * 3;
    let scale_bytes = num_points * 3;
    let rotation_bytes = num_points * (if uses_quaternion_smallest_three { 4 } else { 3 });
    let sh_bytes = num_points * sh_rest_dim * 3;

    let expected_size = 16 + position_bytes + alpha_bytes + color_bytes + scale_bytes + rotation_bytes + sh_bytes;
    if decompressed.len() < expected_size {
        anyhow::bail!(
            "SPZ file too small: expected at least {} bytes, got {}",
            expected_size,
            decompressed.len()
        );
    }

    // Read positions (24-bit fixed point)
    let scale = 1.0 / (1 << fractional_bits) as f32;
    let mut positions = Vec::with_capacity(num_points * 3);
    let mut offset = 16;

    for _ in 0..num_points {
        for _ in 0..3 {
            let mut fixed32 = decompressed[offset] as i32
                | (decompressed[offset + 1] as i32) << 8
                | (decompressed[offset + 2] as i32) << 16;
            // Sign extend from 24th bit
            if fixed32 & 0x800000 != 0 {
                fixed32 |= !0xffffff_i32;
            }
            positions.push(fixed32 as f32 * scale);
            offset += 3;
        }
    }

    // Read alphas and convert via inverse sigmoid
    let mut raw_opacities = Vec::with_capacity(num_points);
    for _ in 0..num_points {
        let alpha = decompressed[offset] as f32 / 255.0;
        raw_opacities.push(inv_sigmoid(alpha));
        offset += 1;
    }

    // Read colors (DC)
    let mut dc_colors = Vec::with_capacity(num_points * 3);
    for _ in 0..num_points {
        for _ in 0..3 {
            let color = ((decompressed[offset] as f32 / 255.0) - 0.5) / COLOR_SCALE;
            dc_colors.push(color);
            offset += 1;
        }
    }

    // Read scales
    let mut log_scales = Vec::with_capacity(num_points * 3);
    for _ in 0..num_points {
        for _ in 0..3 {
            let s = decompressed[offset] as f32 / 16.0 - 10.0;
            log_scales.push(s);
            offset += 1;
        }
    }

    // Read rotations
    let mut rotations = Vec::with_capacity(num_points * 4);
    if uses_quaternion_smallest_three {
        for _ in 0..num_points {
            let r_bytes = &decompressed[offset..offset + 4];
            let q = unpack_quaternion_smallest_three(r_bytes);
            // q is [x, y, z, w], convert to [w, x, y, z] for Splats (brush internal format)
            rotations.push(q[3]); // w
            rotations.push(q[0]); // x
            rotations.push(q[1]); // y
            rotations.push(q[2]); // z
            offset += 4;
        }
    } else {
        // Legacy "first three" format (version 1-2)
        for _ in 0..num_points {
            let x = (decompressed[offset] as f32 / 127.5) - 1.0;
            let y = (decompressed[offset + 1] as f32 / 127.5) - 1.0;
            let z = (decompressed[offset + 2] as f32 / 127.5) - 1.0;
            let sum_sq = x * x + y * y + z * z;
            let w = (1.0 - sum_sq).max(0.0).sqrt();
            rotations.push(x);
            rotations.push(y);
            rotations.push(z);
            rotations.push(w);
            offset += 3;
        }
    }

    // Read SH coefficients and combine with DC
    // sh_coeffs layout: [N, coeffs_per_channel, 3] flattened
    let mut sh_coeffs = Vec::with_capacity(num_points * coeffs_per_channel * 3);
    for i in 0..num_points {
        // DC coefficients first
        sh_coeffs.push(dc_colors[i * 3 + 0]);
        sh_coeffs.push(dc_colors[i * 3 + 1]);
        sh_coeffs.push(dc_colors[i * 3 + 2]);

        // Rest SH coefficients
        for _ in 0..sh_rest_dim {
            for _ in 0..3 {
                let val = unquantize_sh(decompressed[offset]);
                sh_coeffs.push(val);
                offset += 1;
            }
        }
    }

    Ok(DecodedSpz {
        num_points,
        sh_degree,
        positions,
        log_scales,
        rotations,
        raw_opacities,
        sh_coeffs,
    })
}

/// Decodes an SPZ file into a Splats object.
pub fn spz_to_splat<B: Backend>(data: &[u8], device: &B::Device) -> anyhow::Result<Splats<B>> {
    let decoded = spz_to_raw(data)?;

    Ok(Splats::from_raw(
        decoded.positions,
        decoded.rotations,
        decoded.log_scales,
        decoded.sh_coeffs,
        decoded.raw_opacities,
        device,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{LittleEndian, ReadBytesExt};
    use flate2::read::GzDecoder;
    use std::io::{Cursor, Read};

    #[test]
    fn test_quantize_sh() {
        // Test that quantize_sh matches the reference implementation
        // quantize_sh(0.0, 1) should give 128
        assert_eq!(quantize_sh(0.0, 1), 128);

        // quantize_sh with bucket_size=8 (5-bit precision)
        // Values should be rounded to nearest multiple of 8
        assert_eq!(quantize_sh(0.0, 8), 128);

        // quantize_sh with bucket_size=16 (4-bit precision)
        assert_eq!(quantize_sh(0.0, 16), 128);

        // Test edge cases
        assert_eq!(quantize_sh(1.0, 1), 255); // 1.0 * 128 + 128 = 256, clamped to 255
        assert_eq!(quantize_sh(-1.0, 1), 0); // -1.0 * 128 + 128 = 0
    }

    #[test]
    fn test_pack_quaternion() {
        // Test identity quaternion [0, 0, 0, 1] in [x, y, z, w] order
        // w=1 is the largest component, which is at index 3 in [x,y,z,w]
        let mut r = [0u8; 4];
        pack_quaternion_smallest_three(&mut r, [0.0, 0.0, 0.0, 1.0]);
        let comp = r[0] as u32 | (r[1] as u32) << 8 | (r[2] as u32) << 16 | (r[3] as u32) << 24;
        let i_largest = comp >> 30;
        // i_largest should be 3 (w is at index 3 in [x,y,z,w])
        assert_eq!(i_largest, 3);
    }

    #[test]
    fn test_sh_rest_dim() {
        assert_eq!(sh_rest_dim_for_degree(0), 0);
        assert_eq!(sh_rest_dim_for_degree(1), 3);
        assert_eq!(sh_rest_dim_for_degree(2), 8);
        assert_eq!(sh_rest_dim_for_degree(3), 15);
    }

    /// Parsed SPZ header for testing
    #[derive(Debug, PartialEq)]
    struct SpzHeader {
        magic: u32,
        version: u32,
        num_points: u32,
        sh_degree: u8,
        fractional_bits: u8,
        flags: u8,
        reserved: u8,
    }

    /// Parse an SPZ file and return its header and decompressed data
    fn parse_spz(data: &[u8]) -> (SpzHeader, Vec<u8>) {
        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed).expect("Failed to decompress SPZ");

        let mut cursor = Cursor::new(&decompressed);
        let header = SpzHeader {
            magic: cursor.read_u32::<LittleEndian>().unwrap(),
            version: cursor.read_u32::<LittleEndian>().unwrap(),
            num_points: cursor.read_u32::<LittleEndian>().unwrap(),
            sh_degree: cursor.read_u8().unwrap(),
            fractional_bits: cursor.read_u8().unwrap(),
            flags: cursor.read_u8().unwrap(),
            reserved: cursor.read_u8().unwrap(),
        };

        (header, decompressed)
    }

    #[test]
    fn test_spz_header_format() {
        // Create a minimal test to verify header format
        let (header, _) = parse_spz(&create_test_spz_minimal());
        assert_eq!(header.magic, MAGIC);
        assert_eq!(header.version, VERSION);
    }

    /// Create a minimal SPZ file for testing header format
    fn create_test_spz_minimal() -> Vec<u8> {
        let mut uncompressed = Vec::new();
        uncompressed.write_u32::<LittleEndian>(MAGIC).unwrap();
        uncompressed.write_u32::<LittleEndian>(VERSION).unwrap();
        uncompressed.write_u32::<LittleEndian>(0).unwrap(); // num_points
        uncompressed.write_u8(0).unwrap(); // sh_degree
        uncompressed.write_u8(12).unwrap(); // fractional_bits
        uncompressed.write_u8(0).unwrap(); // flags
        uncompressed.write_u8(0).unwrap(); // reserved

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&uncompressed).unwrap();
        encoder.finish().unwrap()
    }

    fn make_test_splats<B: Backend>(include_sh: bool, device: &B::Device) -> Splats<B> {
        let positions = vec![0.0, 0.1, -0.2, 0.3, 0.4, 0.5];
        let scales = vec![-3.0, -2.0, -1.5, -1.0, 0.0, 0.1];
        // Brush stores rotations in [w, x, y, z] order
        // These are arbitrary test quaternions (not necessarily valid rotations without normalization)
        let rotations = vec![0.5, -0.5, 0.2, 1.0, 0.5, 0.1, -0.4, -0.3];
        let alphas = vec![-1.0, 1.0]; // Raw opacities

        let num_points = 2;
        let mut sh_coeffs = Vec::new();

        let colors = vec![
            -1.0, 0.0, 1.0,
            -0.5, 0.5, 0.1
        ];

        let python_sh: Vec<f32> = if include_sh {
            (0..90).map(|i| i as f32 / 45.0 - 1.0).collect()
        } else {
            Vec::new()
        };

        for i in 0..num_points {
            // DC
            sh_coeffs.push(colors[i * 3 + 0]);
            sh_coeffs.push(colors[i * 3 + 1]);
            sh_coeffs.push(colors[i * 3 + 2]);

            if include_sh {
                // Rest coeffs
                let base = i * 45;
                for k in 0..45 {
                    sh_coeffs.push(python_sh[base + k]);
                }
            }
        }

        Splats::from_raw(
            positions,
            rotations,
            scales,
            sh_coeffs,
            alphas,
            device,
        )
    }

    #[tokio::test]
    async fn test_save_load_packed_format() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let splats = make_test_splats::<burn::backend::NdArray>(true, &device);

        let spz_data = splat_to_spz(splats).await.expect("Failed to save SPZ");

        let (header, decompressed) = parse_spz(&spz_data);
        assert_eq!(header.num_points, 2);
        assert_eq!(header.sh_degree, 3);

        // Verify positions
        let scale_pos = (1 << header.fractional_bits) as f32;
        let mut offset = 16;
        let pos_tol = 1.0 / 2048.0;

        let expected_pos = [0.0, 0.1, -0.2, 0.3, 0.4, 0.5];
        for i in 0..2 {
            for j in 0..3 {
                let mut val_i32 = decompressed[offset] as i32
                    | (decompressed[offset + 1] as i32) << 8
                    | (decompressed[offset + 2] as i32) << 16;
                // Sign extend from 24th bit
                if val_i32 & 0x800000 != 0 {
                    val_i32 |= !0xffffff;
                }
                let val = val_i32 as f32 / scale_pos;
                offset += 3;
                assert!(
                    (val - expected_pos[i * 3 + j]).abs() < pos_tol,
                    "Position mismatch at {}: {} vs {}",
                    i * 3 + j,
                    val,
                    expected_pos[i * 3 + j]
                );
            }
        }

        // Verify alphas
        let alpha_tol = 0.01;
        let expected_raw_alphas: [f32; 2] = [-1.0, 1.0];
        for i in 0..2 {
            let val = decompressed[offset] as f32 / 255.0;
            offset += 1;
            let expected_alpha = 1.0 / (1.0 + (-expected_raw_alphas[i]).exp());
            assert!(
                (val - expected_alpha).abs() < alpha_tol,
                "Alpha mismatch at {}: {} vs {}",
                i,
                val,
                expected_alpha
            );
        }

        // Verify colors (DC)
        let expected_colors = [-1.0, 0.0, 1.0, -0.5, 0.5, 0.1];
        for i in 0..6 {
            let val = (decompressed[offset] as f32 - 127.5) / (COLOR_SCALE * 255.0);
            offset += 1;
            assert!(
                (val - expected_colors[i]).abs() < 0.05,
                "Color mismatch at {}: {} vs {}",
                i,
                val,
                expected_colors[i]
            );
        }

        // Verify scales
        let scale_tol = 1.0 / 16.0;
        let expected_scales = [-3.0, -2.0, -1.5, -1.0, 0.0, 0.1];
        for i in 0..6 {
            let val = decompressed[offset] as f32 / 16.0 - 10.0;
            offset += 1;
            assert!(
                (val - expected_scales[i]).abs() < scale_tol,
                "Scale mismatch at {}: {} vs {}",
                i,
                val,
                expected_scales[i]
            );
        }

        // Verify rotations
        offset += 2 * 4;

        // Verify SH
        let expected_sh: Vec<f32> = (0..90).map(|i| i as f32 / 45.0 - 1.0).collect();
        let mut sh_idx = 0;
        for _p in 0..2 {
            for k in 1..16 {
                // 15 rest coeffs
                let _bucket_size = if k <= 3 { 8 } else { 16 };
                for _c in 0..3 {
                    let val_byte = decompressed[offset];
                    offset += 1;
                    let val = (val_byte as f32 - 128.0) / 128.0;
                    let expected = expected_sh[sh_idx];
                    sh_idx += 1;
                    let tol = if k <= 3 { 0.04 } else { 0.07 };
                    assert!(
                        (val - expected).abs() < tol,
                        "SH mismatch at point {}, coeff {}, channel {}: {} vs {}",
                        _p,
                        k,
                        _c,
                        val,
                        expected
                    );
                }
            }
        }
    }

    #[tokio::test]
    async fn test_sh_encoding_for_zeros_and_edges() {
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;

        let positions = vec![0.0, 0.0, 0.0];
        let scales = vec![0.0, 0.0, 0.0];
        let rotations = vec![0.0, 0.0, 0.0, 1.0];
        let raw_opacities = vec![-10.0]; // Effectively 0 alpha

        let colors = vec![0.0, 0.0, 0.0];
        let sh_rest = vec![-0.01, 0.0, 0.01, -1.0, -0.99, -0.95, 0.95, 0.99, 1.0];

        let mut sh_coeffs = Vec::new();
        sh_coeffs.extend_from_slice(&colors);
        sh_coeffs.extend_from_slice(&sh_rest);

        let splats = Splats::<burn::backend::NdArray>::from_raw(
            positions,
            rotations,
            scales,
            sh_coeffs,
            raw_opacities,
            &device,
        );

        let spz_data = splat_to_spz(splats).await.expect("Failed to save SPZ");
        let (header, decompressed) = parse_spz(&spz_data);

        assert_eq!(header.num_points, 1);
        assert_eq!(header.sh_degree, 1);

        let num_points = 1;
        let positions_end = 16 + num_points * 9;
        let alphas_end = positions_end + num_points;
        let colors_end = alphas_end + num_points * 3;
        let scales_end = colors_end + num_points * 3;
        let rotations_end = scales_end + num_points * 4;
        let mut offset = rotations_end;

        let expected_sh = [0.0, 0.0, 0.0, -1.0, -1.0, -0.9375, 0.9375, 0.9922, 0.9922];

        for i in 0..9 {
            let val_byte = decompressed[offset];
            offset += 1;
            let val = (val_byte as f32 - 128.0) / 128.0;
            assert!(
                (val - expected_sh[i]).abs() < 2e-2,
                "SH mismatch at {}: {} vs {}",
                i,
                val,
                expected_sh[i]
            );
        }
    }

    #[test]
    fn test_quaternion_normalization_during_packing() {
        // Test that non-normalized quaternions get normalized during packing
        // and that the packed result represents the same rotation.

        // Non-normalized quaternion [w, x, y, z] = [2, 3, 4, 5]
        let non_normalized = [2.0f32, 3.0, 4.0, 5.0];
        let norm = (non_normalized[0].powi(2)
            + non_normalized[1].powi(2)
            + non_normalized[2].powi(2)
            + non_normalized[3].powi(2))
        .sqrt();
        let normalized = [
            non_normalized[0] / norm,
            non_normalized[1] / norm,
            non_normalized[2] / norm,
            non_normalized[3] / norm,
        ];

        let mut r1 = [0u8; 4];
        let mut r2 = [0u8; 4];

        pack_quaternion_smallest_three(&mut r1, non_normalized);
        pack_quaternion_smallest_three(&mut r2, normalized);

        // Both should pack to the same result since normalization happens inside
        assert_eq!(r1, r2, "Non-normalized and normalized quaternions should pack identically");

        // Also test that negated quaternion (representing same rotation) produces valid output
        let negated = [-normalized[0], -normalized[1], -normalized[2], -normalized[3]];
        let mut r3 = [0u8; 4];
        pack_quaternion_smallest_three(&mut r3, negated);

        // The packed values may differ but both represent valid rotations
        // Just verify it doesn't panic and produces a valid packed value
        let comp = r3[0] as u32 | (r3[1] as u32) << 8 | (r3[2] as u32) << 16 | (r3[3] as u32) << 24;
        let i_largest = comp >> 30;
        assert!(i_largest < 4, "i_largest should be 0-3");
    }

    #[test]
    fn test_compression_precision_validation() {
        // Test that compression maintains expected precision levels
        // Based on Python test_compression_precision_validation

        // Position precision: 12-bit fractional (1/4096 resolution)
        let pos_values = [1.0f32, -1.0, 0.5, 0.0, -0.25, 0.125];
        let fractional_bits = 12;
        let scale_pos = (1 << fractional_bits) as f32;

        for pos in pos_values {
            let fixed32 = (pos * scale_pos).round() as i32;
            let reconstructed = fixed32 as f32 / scale_pos;
            assert!(
                (reconstructed - pos).abs() < 1.0 / 2048.0,
                "Position precision failure: {} vs {}",
                reconstructed,
                pos
            );
        }

        // Scale precision: 5-bit (1/32 resolution)
        let scale_values = [1.0f32, -1.0, 0.5, -3.0, 0.0, 5.0];
        for s in scale_values {
            let packed = to_uint8((s + 10.0) * 16.0);
            let reconstructed = packed as f32 / 16.0 - 10.0;
            assert!(
                (reconstructed - s).abs() < 1.0 / 16.0,
                "Scale precision failure: {} vs {}",
                reconstructed,
                s
            );
        }

        // Alpha precision: 8-bit
        let alpha_values = [0.0f32, 0.5, 1.0, 0.25, 0.75];
        for a in alpha_values {
            let packed = to_uint8(a * 255.0);
            let reconstructed = packed as f32 / 255.0;
            assert!(
                (reconstructed - a).abs() < 0.01,
                "Alpha precision failure: {} vs {}",
                reconstructed,
                a
            );
        }

        // SH precision: 5-bit for degree-1, 4-bit for rest
        let sh_values = [0.0f32, 0.5, -0.5, 0.25, -0.25];
        for sh in sh_values {
            // 5-bit precision (bucket_size = 8)
            let packed_5bit = quantize_sh(sh, 8);
            let reconstructed_5bit = (packed_5bit as f32 - 128.0) / 128.0;
            let epsilon_5bit = 2.0 / 64.0 + 0.5 / 255.0;
            assert!(
                (reconstructed_5bit - sh).abs() < epsilon_5bit,
                "SH 5-bit precision failure: {} vs {}",
                reconstructed_5bit,
                sh
            );

            // 4-bit precision (bucket_size = 16)
            let packed_4bit = quantize_sh(sh, 16);
            let reconstructed_4bit = (packed_4bit as f32 - 128.0) / 128.0;
            let epsilon_4bit = 2.0 / 32.0 + 0.5 / 255.0;
            assert!(
                (reconstructed_4bit - sh).abs() < epsilon_4bit,
                "SH 4-bit precision failure: {} vs {}",
                reconstructed_4bit,
                sh
            );
        }
    }

    #[test]
    fn test_pack_quaternion_various_orientations() {
        // Test packing quaternions with different largest components
        // Note: pack function expects [x, y, z, w] order

        // Identity: [0, 0, 0, 1] in [x,y,z,w] - w is largest at index 3
        let mut r = [0u8; 4];
        pack_quaternion_smallest_three(&mut r, [0.0, 0.0, 0.0, 1.0]);
        let comp = r[0] as u32 | (r[1] as u32) << 8 | (r[2] as u32) << 16 | (r[3] as u32) << 24;
        let i_largest = (comp >> 30) as usize;
        // i_largest should be 3 (w at index 3)
        assert_eq!(i_largest, 3);

        // 90 degree rotation around X: in [x,y,z,w] = [sin(45), 0, 0, cos(45)] = [0.707, 0, 0, 0.707]
        let sqrt2_2 = std::f32::consts::FRAC_1_SQRT_2;
        pack_quaternion_smallest_three(&mut r, [sqrt2_2, 0.0, 0.0, sqrt2_2]);
        // Should not panic and produce valid output

        // 90 degree rotation around Y: [x,y,z,w] = [0, sin(45), 0, cos(45)]
        pack_quaternion_smallest_three(&mut r, [0.0, sqrt2_2, 0.0, sqrt2_2]);

        // 90 degree rotation around Z: [x,y,z,w] = [0, 0, sin(45), cos(45)]
        pack_quaternion_smallest_three(&mut r, [0.0, 0.0, sqrt2_2, sqrt2_2]);

        // Quaternion where x is largest: [0.9, 0.2, 0.3, 0.1] in [x,y,z,w]
        pack_quaternion_smallest_three(&mut r, [0.9, 0.2, 0.3, 0.1]);
        let comp = r[0] as u32 | (r[1] as u32) << 8 | (r[2] as u32) << 16 | (r[3] as u32) << 24;
        let stored = (comp >> 30) as usize;
        // x is largest at index 0
        assert_eq!(stored, 0);

        // Quaternion where y is largest: [0.2, 0.9, 0.3, 0.1] in [x,y,z,w]
        pack_quaternion_smallest_three(&mut r, [0.2, 0.9, 0.3, 0.1]);
        let comp = r[0] as u32 | (r[1] as u32) << 8 | (r[2] as u32) << 16 | (r[3] as u32) << 24;
        let stored = (comp >> 30) as usize;
        // y is largest at index 1
        assert_eq!(stored, 1);

        // Quaternion where z is largest: [0.2, 0.3, 0.9, 0.1] in [x,y,z,w]
        pack_quaternion_smallest_three(&mut r, [0.2, 0.3, 0.9, 0.1]);
        let comp = r[0] as u32 | (r[1] as u32) << 8 | (r[2] as u32) << 16 | (r[3] as u32) << 24;
        let stored = (comp >> 30) as usize;
        // z is largest at index 2
        assert_eq!(stored, 2);
    }

    #[tokio::test]
    async fn test_large_splat_serialization() {
        // Test with a larger number of points to verify the format handles many points
        // Based on Python test_save_load_packed_format_large_splat

        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let num_points = 1000;

        // Generate deterministic test data using simple formulas
        let mut positions = Vec::with_capacity(num_points * 3);
        let mut scales = Vec::with_capacity(num_points * 3);
        let mut rotations = Vec::with_capacity(num_points * 4);
        let mut raw_opacities = Vec::with_capacity(num_points);
        let mut sh_coeffs = Vec::with_capacity(num_points * 3); // Just DC, no SH rest

        for i in 0..num_points {
            let t = i as f32 / num_points as f32;

            // Positions in range [-1, 1]
            positions.push((t * 2.0 - 1.0) * 0.5);
            positions.push(((t * 3.0) % 1.0) * 2.0 - 1.0);
            positions.push(((t * 7.0) % 1.0) * 2.0 - 1.0);

            // Scales in range [-5, 5]
            scales.push((t * 10.0 - 5.0) * 0.5);
            scales.push(((t * 2.0) % 1.0) * 10.0 - 5.0);
            scales.push(((t * 5.0) % 1.0) * 10.0 - 5.0);

            // Random-ish quaternions (will be normalized by pack function)
            let qx = (t * 13.0).sin();
            let qy = (t * 17.0).cos();
            let qz = (t * 23.0).sin();
            let qw = (t * 29.0).cos();
            rotations.push(qx);
            rotations.push(qy);
            rotations.push(qz);
            rotations.push(qw);

            // Raw opacities (will be converted via sigmoid)
            raw_opacities.push(t * 4.0 - 2.0);

            // DC colors
            sh_coeffs.push(t);
            sh_coeffs.push(1.0 - t);
            sh_coeffs.push((t * 2.0) % 1.0);
        }

        let splats = Splats::<burn::backend::NdArray>::from_raw(
            positions.clone(),
            rotations,
            scales.clone(),
            sh_coeffs,
            raw_opacities.clone(),
            &device,
        );

        let spz_data = splat_to_spz(splats).await.expect("Failed to save large SPZ");
        let (header, decompressed) = parse_spz(&spz_data);

        assert_eq!(header.num_points, num_points as u32);
        assert_eq!(header.sh_degree, 0); // No SH rest coefficients

        // Verify decompressed size matches expected
        let expected_size = 16 // header
            + num_points * 9  // positions (3 * 3 bytes each)
            + num_points      // alphas
            + num_points * 3  // colors
            + num_points * 3  // scales
            + num_points * 4; // rotations
        assert_eq!(decompressed.len(), expected_size);

        // Spot check some positions
        let scale_pos = (1 << header.fractional_bits) as f32;
        let pos_tol = 1.0 / 2048.0;

        for check_idx in [0, 100, 500, 999] {
            let offset = 16 + check_idx * 9;
            for j in 0..3 {
                let byte_offset = offset + j * 3;
                let mut val_i32 = decompressed[byte_offset] as i32
                    | (decompressed[byte_offset + 1] as i32) << 8
                    | (decompressed[byte_offset + 2] as i32) << 16;
                if val_i32 & 0x800000 != 0 {
                    val_i32 |= !0xffffff;
                }
                let val = val_i32 as f32 / scale_pos;
                let expected = positions[check_idx * 3 + j];
                assert!(
                    (val - expected).abs() < pos_tol,
                    "Position mismatch at point {}, component {}: {} vs {}",
                    check_idx,
                    j,
                    val,
                    expected
                );
            }
        }

        // Spot check some alphas
        let alpha_offset = 16 + num_points * 9;
        let alpha_tol = 0.01;
        for check_idx in [0, 100, 500, 999] {
            let val = decompressed[alpha_offset + check_idx] as f32 / 255.0;
            let expected_alpha = 1.0 / (1.0 + (-raw_opacities[check_idx]).exp());
            assert!(
                (val - expected_alpha).abs() < alpha_tol,
                "Alpha mismatch at point {}: {} vs {}",
                check_idx,
                val,
                expected_alpha
            );
        }

        // Spot check some scales
        let scales_offset = 16 + num_points * 9 + num_points + num_points * 3;
        let scale_tol = 1.0 / 16.0;
        for check_idx in [0, 100, 500, 999] {
            for j in 0..3 {
                let val = decompressed[scales_offset + check_idx * 3 + j] as f32 / 16.0 - 10.0;
                let expected = scales[check_idx * 3 + j];
                assert!(
                    (val - expected).abs() < scale_tol,
                    "Scale mismatch at point {}, component {}: {} vs {}",
                    check_idx,
                    j,
                    val,
                    expected
                );
            }
        }
    }

    #[tokio::test]
    async fn test_single_point_splat() {
        // Test serializing a single point splat
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;

        let splats = Splats::<burn::backend::NdArray>::from_raw(
            vec![0.5, -0.5, 0.25],      // single position
            vec![1.0, 0.0, 0.0, 0.0],   // identity rotation [w,x,y,z] format
            vec![0.1, 0.2, 0.3],        // scales
            vec![0.5, 0.5, 0.5],        // DC color only
            vec![0.0],                   // raw opacity
            &device,
        );

        let spz_data = splat_to_spz(splats).await.expect("Failed to save single point SPZ");
        let (header, decompressed) = parse_spz(&spz_data);

        assert_eq!(header.num_points, 1);
        assert_eq!(header.sh_degree, 0);
        assert_eq!(header.magic, MAGIC);
        assert_eq!(header.version, VERSION);

        // Verify expected size: header + 1 point worth of data
        let expected_size = 16  // header
            + 1 * 9   // positions
            + 1       // alphas
            + 1 * 3   // colors
            + 1 * 3   // scales
            + 1 * 4;  // rotations
        assert_eq!(decompressed.len(), expected_size);

        // Verify position is correctly encoded
        let scale_pos = (1 << header.fractional_bits) as f32;
        let pos_tol = 1.0 / 2048.0;
        let expected_pos = [0.5, -0.5, 0.25];

        for j in 0..3 {
            let offset = 16 + j * 3;
            let mut val_i32 = decompressed[offset] as i32
                | (decompressed[offset + 1] as i32) << 8
                | (decompressed[offset + 2] as i32) << 16;
            if val_i32 & 0x800000 != 0 {
                val_i32 |= !0xffffff;
            }
            let val = val_i32 as f32 / scale_pos;
            assert!(
                (val - expected_pos[j]).abs() < pos_tol,
                "Position mismatch at component {}: {} vs {}",
                j,
                val,
                expected_pos[j]
            );
        }
    }

    #[test]
    fn test_to_uint8_edge_cases() {
        // Test to_uint8 with edge cases
        assert_eq!(to_uint8(0.0), 0);
        assert_eq!(to_uint8(255.0), 255);
        assert_eq!(to_uint8(127.5), 128); // Rounds to nearest
        assert_eq!(to_uint8(127.4), 127);
        assert_eq!(to_uint8(127.6), 128);

        // Test clamping
        assert_eq!(to_uint8(-100.0), 0);
        assert_eq!(to_uint8(1000.0), 255);
        assert_eq!(to_uint8(f32::NEG_INFINITY), 0);
        assert_eq!(to_uint8(f32::INFINITY), 255);
    }

    #[test]
    fn test_quantize_sh_edge_cases() {
        // Test quantize_sh with various edge cases

        // Values at boundaries
        assert_eq!(quantize_sh(1.0, 1), 255);
        assert_eq!(quantize_sh(-1.0, 1), 0);
        assert_eq!(quantize_sh(0.5, 1), 192); // 0.5 * 128 + 128 = 192
        assert_eq!(quantize_sh(-0.5, 1), 64); // -0.5 * 128 + 128 = 64

        // Test with bucket_size = 8 (5-bit)
        // Should round to nearest multiple of 8
        assert_eq!(quantize_sh(0.0, 8), 128);
        // 0.1 * 128 + 128 = 140.8, rounded = 141, nearest multiple of 8 = 144
        let q = quantize_sh(0.1, 8);
        assert!(q % 8 == 0 || q == 255 || q == 0);

        // Test with bucket_size = 16 (4-bit)
        assert_eq!(quantize_sh(0.0, 16), 128);
        let q = quantize_sh(0.2, 16);
        assert!(q % 16 == 0 || q == 255 || q == 0);

        // Test clamping at extremes
        assert_eq!(quantize_sh(2.0, 1), 255); // Should clamp to 255
        assert_eq!(quantize_sh(-2.0, 1), 0); // Should clamp to 0
    }

    #[test]
    fn test_unpack_quaternion_smallest_three() {
        // Test that pack followed by unpack recovers the original quaternion
        // Both pack and unpack use [x, y, z, w] order
        let test_quaternions: [[f32; 4]; 6] = [
            [0.0, 0.0, 0.0, 1.0],           // identity (w largest at index 3)
            [1.0, 0.0, 0.0, 0.0],           // x largest at index 0
            [0.0, 1.0, 0.0, 0.0],           // y largest at index 1
            [0.0, 0.0, 1.0, 0.0],           // z largest at index 2
            [0.5, 0.5, 0.5, 0.5],           // uniform
            [0.7071, 0.0, 0.0, 0.7071],     // 90 deg around x: [sin(45), 0, 0, cos(45)]
        ];

        for q_in in test_quaternions {
            // Normalize
            let norm = (q_in[0] * q_in[0] + q_in[1] * q_in[1] + q_in[2] * q_in[2] + q_in[3] * q_in[3]).sqrt();
            let q_normalized = [q_in[0] / norm, q_in[1] / norm, q_in[2] / norm, q_in[3] / norm];

            let mut packed = [0u8; 4];
            pack_quaternion_smallest_three(&mut packed, q_normalized);
            let q_out = unpack_quaternion_smallest_three(&packed);

            // Check that the quaternions represent the same rotation (dot product close to 1)
            let dot = (q_normalized[0] * q_out[0]
                + q_normalized[1] * q_out[1]
                + q_normalized[2] * q_out[2]
                + q_normalized[3] * q_out[3])
            .abs();

            assert!(
                dot > 0.99,
                "Quaternion round-trip failed: {:?} -> {:?}, dot = {}",
                q_normalized,
                q_out,
                dot
            );
        }
    }

    #[test]
    fn test_inv_sigmoid() {
        // Test that sigmoid followed by inv_sigmoid recovers original
        let test_values = [-5.0f32, -1.0, 0.0, 1.0, 5.0];

        for raw in test_values {
            let alpha = 1.0 / (1.0 + (-raw).exp());
            let recovered = inv_sigmoid(alpha);
            assert!(
                (recovered - raw).abs() < 0.01,
                "inv_sigmoid round-trip failed: {} -> {} -> {}",
                raw,
                alpha,
                recovered
            );
        }
    }

    #[tokio::test]
    async fn test_encode_decode_roundtrip_no_sh() {
        // Test encode -> decode round-trip with no SH coefficients (degree 0)
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;

        let positions = vec![0.5, -0.5, 0.25, -0.25, 0.1, -0.1];
        let scales = vec![-3.0, -2.0, -1.0, 1.0, 2.0, 3.0];
        // Brush stores rotations in [w, x, y, z] order
        let rotations = vec![
            1.0, 0.0, 0.0, 0.0, // identity: w=1, x=y=z=0
            0.5, 0.5, 0.5, 0.5, // uniform
        ];
        let raw_opacities = vec![-1.0, 1.0];
        let sh_coeffs = vec![
            0.5, 0.3, 0.1, // DC for point 0
            -0.2, 0.4, -0.3, // DC for point 1
        ];

        let splats = Splats::<burn::backend::NdArray>::from_raw(
            positions.clone(),
            rotations.clone(),
            scales.clone(),
            sh_coeffs.clone(),
            raw_opacities.clone(),
            &device,
        );

        let spz_data = splat_to_spz(splats).await.expect("Failed to encode SPZ");
        let decoded = spz_to_raw(&spz_data).expect("Failed to decode SPZ");

        assert_eq!(decoded.num_points, 2);
        assert_eq!(decoded.sh_degree, 0);

        // Check positions (tolerance: ~1/2048)
        let pos_tol = 1.0 / 2048.0;
        for (i, (&expected, &actual)) in positions.iter().zip(decoded.positions.iter()).enumerate() {
            assert!(
                (expected - actual).abs() < pos_tol,
                "Position mismatch at {}: {} vs {}",
                i,
                expected,
                actual
            );
        }

        // Check scales (tolerance: 1/16)
        let scale_tol = 1.0 / 16.0;
        for (i, (&expected, &actual)) in scales.iter().zip(decoded.log_scales.iter()).enumerate() {
            assert!(
                (expected - actual).abs() < scale_tol,
                "Scale mismatch at {}: {} vs {}",
                i,
                expected,
                actual
            );
        }

        // Check alphas via comparing sigmoid values (tolerance: 1%)
        let alpha_tol = 0.01;
        for (i, (&expected_raw, &actual_raw)) in
            raw_opacities.iter().zip(decoded.raw_opacities.iter()).enumerate()
        {
            let expected_alpha = 1.0 / (1.0 + (-expected_raw).exp());
            let actual_alpha = 1.0 / (1.0 + (-actual_raw).exp());
            assert!(
                (expected_alpha - actual_alpha).abs() < alpha_tol,
                "Alpha mismatch at {}: {} vs {} (raw: {} vs {})",
                i,
                expected_alpha,
                actual_alpha,
                expected_raw,
                actual_raw
            );
        }

        // Check DC colors (tolerance: ~5%)
        let color_tol = 0.15;
        for (i, (&expected, &actual)) in sh_coeffs.iter().zip(decoded.sh_coeffs.iter()).enumerate() {
            assert!(
                (expected - actual).abs() < color_tol,
                "DC color mismatch at {}: {} vs {}",
                i,
                expected,
                actual
            );
        }

        // Check rotations (compare as quaternion rotations)
        // Both expected and actual are in brush's [w, x, y, z] format
        for p in 0..2 {
            let expected_q = [
                rotations[p * 4 + 0], // w
                rotations[p * 4 + 1], // x
                rotations[p * 4 + 2], // y
                rotations[p * 4 + 3], // z
            ];
            let actual_q = [
                decoded.rotations[p * 4 + 0], // w
                decoded.rotations[p * 4 + 1], // x
                decoded.rotations[p * 4 + 2], // y
                decoded.rotations[p * 4 + 3], // z
            ];

            // Normalize both
            let norm_e =
                (expected_q[0].powi(2) + expected_q[1].powi(2) + expected_q[2].powi(2) + expected_q[3].powi(2)).sqrt();
            let norm_a =
                (actual_q[0].powi(2) + actual_q[1].powi(2) + actual_q[2].powi(2) + actual_q[3].powi(2)).sqrt();

            let dot = ((expected_q[0] / norm_e) * (actual_q[0] / norm_a)
                + (expected_q[1] / norm_e) * (actual_q[1] / norm_a)
                + (expected_q[2] / norm_e) * (actual_q[2] / norm_a)
                + (expected_q[3] / norm_e) * (actual_q[3] / norm_a))
            .abs();

            assert!(
                dot > 0.99,
                "Rotation mismatch at point {}: expected {:?}, got {:?}, dot = {}",
                p,
                expected_q,
                actual_q,
                dot
            );
        }
    }

    #[tokio::test]
    async fn test_encode_decode_roundtrip_with_sh() {
        // Test encode -> decode round-trip with SH degree 3
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;

        let positions = vec![0.1, 0.2, 0.3];
        let scales = vec![0.0, -1.0, 1.0];
        // 90 deg around z in [w,x,y,z] format: w=cos(45°), x=0, y=0, z=sin(45°)
        let rotations = vec![0.7071, 0.0, 0.0, 0.7071];
        let raw_opacities = vec![0.0];

        // SH coeffs: [1 point, 16 coeffs per channel, 3 channels]
        // Generate values in [-0.9, 0.9] to stay within valid range
        let mut sh_coeffs = Vec::new();
        for k in 0..16 {
            for c in 0..3 {
                let val = ((k * 3 + c) as f32 / 48.0) * 1.8 - 0.9;
                sh_coeffs.push(val);
            }
        }

        let splats = Splats::<burn::backend::NdArray>::from_raw(
            positions.clone(),
            rotations.clone(),
            scales.clone(),
            sh_coeffs.clone(),
            raw_opacities.clone(),
            &device,
        );

        let spz_data = splat_to_spz(splats).await.expect("Failed to encode SPZ");
        let decoded = spz_to_raw(&spz_data).expect("Failed to decode SPZ");

        assert_eq!(decoded.num_points, 1);
        assert_eq!(decoded.sh_degree, 3);

        // Check SH coefficients
        // DC has 5% tolerance, SH rest has higher tolerance due to quantization
        for (i, (&expected, &actual)) in sh_coeffs.iter().zip(decoded.sh_coeffs.iter()).enumerate() {
            let k = i / 3; // coefficient index
            let tol = if k == 0 {
                0.15 // DC tolerance
            } else if k <= 3 {
                0.10 // 5-bit quantization
            } else {
                0.15 // 4-bit quantization
            };

            assert!(
                (expected - actual).abs() < tol,
                "SH coeff mismatch at {} (k={}): {} vs {}",
                i,
                k,
                expected,
                actual
            );
        }
    }

    #[tokio::test]
    async fn test_encode_decode_roundtrip_large() {
        // Test round-trip with many points
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;
        let num_points = 100;

        let mut positions = Vec::with_capacity(num_points * 3);
        let mut scales = Vec::with_capacity(num_points * 3);
        let mut rotations = Vec::with_capacity(num_points * 4);
        let mut raw_opacities = Vec::with_capacity(num_points);
        let mut sh_coeffs = Vec::with_capacity(num_points * 3);

        for i in 0..num_points {
            let t = i as f32 / num_points as f32;

            // Positions in [-1, 1]
            positions.push(t * 2.0 - 1.0);
            positions.push((t * 3.0) % 1.0 * 2.0 - 1.0);
            positions.push((t * 7.0) % 1.0 * 2.0 - 1.0);

            // Scales in [-5, 5]
            scales.push(t * 10.0 - 5.0);
            scales.push((t * 2.0) % 1.0 * 10.0 - 5.0);
            scales.push((t * 5.0) % 1.0 * 10.0 - 5.0);

            // Quaternion
            let qx = (t * 13.0).sin();
            let qy = (t * 17.0).cos();
            let qz = (t * 23.0).sin();
            let qw = (t * 29.0).cos();
            rotations.push(qx);
            rotations.push(qy);
            rotations.push(qz);
            rotations.push(qw);

            // Opacity
            raw_opacities.push(t * 4.0 - 2.0);

            // DC
            sh_coeffs.push(t - 0.5);
            sh_coeffs.push(0.5 - t);
            sh_coeffs.push((t * 2.0) % 1.0 - 0.5);
        }

        let splats = Splats::<burn::backend::NdArray>::from_raw(
            positions.clone(),
            rotations.clone(),
            scales.clone(),
            sh_coeffs.clone(),
            raw_opacities.clone(),
            &device,
        );

        let spz_data = splat_to_spz(splats).await.expect("Failed to encode SPZ");
        let decoded = spz_to_raw(&spz_data).expect("Failed to decode SPZ");

        assert_eq!(decoded.num_points, num_points);
        assert_eq!(decoded.sh_degree, 0);

        // Spot check positions
        let pos_tol = 1.0 / 2048.0;
        for i in [0, 10, 50, 99] {
            for j in 0..3 {
                let expected = positions[i * 3 + j];
                let actual = decoded.positions[i * 3 + j];
                assert!(
                    (expected - actual).abs() < pos_tol,
                    "Position mismatch at point {} component {}: {} vs {}",
                    i,
                    j,
                    expected,
                    actual
                );
            }
        }

        // Spot check scales
        let scale_tol = 1.0 / 16.0;
        for i in [0, 10, 50, 99] {
            for j in 0..3 {
                let expected = scales[i * 3 + j];
                let actual = decoded.log_scales[i * 3 + j];
                assert!(
                    (expected - actual).abs() < scale_tol,
                    "Scale mismatch at point {} component {}: {} vs {}",
                    i,
                    j,
                    expected,
                    actual
                );
            }
        }
    }

    #[tokio::test]
    async fn test_spz_to_splat_roundtrip() {
        // Test the full Splats -> SPZ -> Splats round-trip
        let device = burn::backend::ndarray::NdArrayDevice::Cpu;

        let original = Splats::<burn::backend::NdArray>::from_raw(
            vec![0.1, 0.2, 0.3, -0.1, -0.2, -0.3],
            vec![1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5], // [w,x,y,z] format: identity, then uniform
            vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0],
            vec![0.3, 0.4, 0.5, -0.3, -0.4, -0.5],
            vec![0.5, -0.5],
            &device,
        );

        let spz_data = splat_to_spz(original).await.expect("Failed to encode SPZ");
        let reconstructed: Splats<burn::backend::NdArray> =
            spz_to_splat(&spz_data, &device).expect("Failed to decode SPZ");

        assert_eq!(reconstructed.num_splats(), 2);
        assert_eq!(reconstructed.sh_degree(), 0);
    }

    /// Compare two SPZ files and return detailed differences
    fn compare_spz_files(rust_data: &[u8], reference_data: &[u8]) -> Result<(), String> {
        // Decompress both files
        let (rust_header, rust_decompressed) = parse_spz(rust_data);
        let (ref_header, ref_decompressed) = parse_spz(reference_data);
        
        // Compare headers
        if rust_header.magic != ref_header.magic {
            return Err(format!("Magic mismatch: rust={:#x}, ref={:#x}", rust_header.magic, ref_header.magic));
        }
        if rust_header.version != ref_header.version {
            return Err(format!("Version mismatch: rust={}, ref={}", rust_header.version, ref_header.version));
        }
        if rust_header.num_points != ref_header.num_points {
            return Err(format!("NumPoints mismatch: rust={}, ref={}", rust_header.num_points, ref_header.num_points));
        }
        if rust_header.sh_degree != ref_header.sh_degree {
            return Err(format!("SH degree mismatch: rust={}, ref={}", rust_header.sh_degree, ref_header.sh_degree));
        }
        if rust_header.fractional_bits != ref_header.fractional_bits {
            return Err(format!("Fractional bits mismatch: rust={}, ref={}", rust_header.fractional_bits, ref_header.fractional_bits));
        }

        // Compare decompressed content byte-by-byte
        if rust_decompressed.len() != ref_decompressed.len() {
            return Err(format!(
                "Decompressed size mismatch: rust={}, ref={}",
                rust_decompressed.len(),
                ref_decompressed.len()
            ));
        }
        
        // Find first byte mismatch
        for i in 0..rust_decompressed.len() {
            if rust_decompressed[i] != ref_decompressed[i] {
                // Determine which section this byte is in
                let header_size = 16usize;
                let num_points = rust_header.num_points as usize;
                let positions_end = header_size + num_points * 9;
                let alphas_end = positions_end + num_points;
                let colors_end = alphas_end + num_points * 3;
                let scales_end = colors_end + num_points * 3;
                let rotations_end = scales_end + num_points * 4;
                
                let section = if i < header_size { "header" }
                    else if i < positions_end { "positions" }
                    else if i < alphas_end { "alphas" }
                    else if i < colors_end { "colors" }
                    else if i < scales_end { "scales" }
                    else if i < rotations_end { "rotations" }
                    else { "sh_coeffs" };
                
                return Err(format!(
                    "Data mismatch at byte {} (section: {}): rust={:#04x}, ref={:#04x}",
                    i, section, rust_decompressed[i], ref_decompressed[i]
                ));
            }
        }

        // If we reached here, decompressed content is identical
        Ok(())
    }
}
