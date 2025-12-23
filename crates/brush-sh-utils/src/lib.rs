//! Spherical harmonics utilities for Gaussian splatting.
//!
//! This crate provides functions for rotating spherical harmonic coefficients,
//! which is necessary when transforming Gaussian splat scenes (e.g., correcting
//! orientation during export).
//!
//! # SH Coefficient Layout
//!
//! Brush uses real spherical harmonics with the following basis ordering:
//! - Degree 0 (1 coeff): DC term (rotationally invariant)
//! - Degree 1 (3 coeffs): Y₁⁻¹, Y₁⁰, Y₁¹ → proportional to y, z, x
//! - Degree 2 (5 coeffs): Y₂⁻², Y₂⁻¹, Y₂⁰, Y₂¹, Y₂²
//! - Degree 3 (7 coeffs): Y₃⁻³, Y₃⁻², Y₃⁻¹, Y₃⁰, Y₃¹, Y₃², Y₃³
//!
//! The rotation of SH coefficients is performed using rotation matrices derived
//! from the 3D rotation. Each SH band (degree) rotates independently within itself.

use glam::{Mat3, Quat};

/// Compute the 3×3 rotation matrix for degree-1 (l=1) spherical harmonics.
///
/// Uses the formulas from Google's spherical-harmonics library which include
/// the Condon-Shortley phase convention.
pub fn sh_rotation_matrix_l1(rot: Mat3) -> [[f32; 3]; 3] {
    // From Google's spherical-harmonics library:
    // The l=1 SH rotation is a permuted/signed version of the 3D rotation matrix.
    // This includes the Condon-Shortley phase.
    
    let r = rot.to_cols_array_2d();
    // r[col][row], so r[i][j] = R_{ji}
    // We need R_{ij} = r[j][i]
    let rm = |row: usize, col: usize| r[col][row];
    
    [
        [rm(1, 1),  -rm(1, 2), rm(1, 0)],
        [-rm(2, 1), rm(2, 2),  -rm(2, 0)],
        [rm(0, 1),  -rm(0, 2), rm(0, 0)],
    ]
}

/// Compute the 5×5 rotation matrix for degree-2 (l=2) spherical harmonics.
///
/// Uses closed-form formulas from Google's spherical-harmonics library test suite.
/// These formulas include the Condon-Shortley phase convention.
pub fn sh_rotation_matrix_l2(rot: Mat3) -> [[f32; 5]; 5] {
    let r = rot.to_cols_array_2d();
    // r[col][row], so we need r[j][i] to get R_{ij}
    let rm = |row: usize, col: usize| r[col][row];
    
    let s3 = 3.0_f32.sqrt();
    
    // Formulas from Google's spherical-harmonics ClosedFormBands test
    [
        // Row 0 (m' = -2, Y₂⁻² ~ xy)
        [
            rm(0,0)*rm(1,1) + rm(0,1)*rm(1,0),                                    // col 0
            -(rm(0,1)*rm(1,2) + rm(0,2)*rm(1,1)),                                 // col 1
            -(s3/3.0) * (rm(0,0)*rm(1,0) + rm(0,1)*rm(1,1) - 2.0*rm(0,2)*rm(1,2)), // col 2
            -(rm(0,0)*rm(1,2) + rm(0,2)*rm(1,0)),                                 // col 3
            rm(0,0)*rm(1,0) - rm(0,1)*rm(1,1),                                    // col 4
        ],
        // Row 1 (m' = -1, Y₂⁻¹ ~ yz)
        [
            -(rm(1,0)*rm(2,1) + rm(1,1)*rm(2,0)),                                 // col 0
            rm(1,1)*rm(2,2) + rm(1,2)*rm(2,1),                                    // col 1
            (s3/3.0) * (rm(1,0)*rm(2,0) + rm(1,1)*rm(2,1) - 2.0*rm(1,2)*rm(2,2)), // col 2
            rm(1,0)*rm(2,2) + rm(1,2)*rm(2,0),                                    // col 3
            -(rm(1,0)*rm(2,0) - rm(1,1)*rm(2,1)),                                 // col 4
        ],
        // Row 2 (m' = 0, Y₂⁰ ~ 3z²-1)
        [
            -(s3/3.0) * (rm(0,0)*rm(0,1) + rm(1,0)*rm(1,1) - 2.0*rm(2,0)*rm(2,1)), // col 0
            (s3/3.0) * (rm(0,1)*rm(0,2) + rm(1,1)*rm(1,2) - 2.0*rm(2,1)*rm(2,2)),  // col 1
            -0.5 * (1.0 - 3.0*rm(2,2)*rm(2,2)),                                    // col 2
            (s3/3.0) * (rm(0,0)*rm(0,2) + rm(1,0)*rm(1,2) - 2.0*rm(2,0)*rm(2,2)),  // col 3
            (s3/6.0) * (-(rm(0,0)*rm(0,0)) + rm(0,1)*rm(0,1) - rm(1,0)*rm(1,0) + rm(1,1)*rm(1,1) + 2.0*rm(2,0)*rm(2,0) - 2.0*rm(2,1)*rm(2,1)), // col 4
        ],
        // Row 3 (m' = 1, Y₂¹ ~ xz)
        [
            -(rm(0,0)*rm(2,1) + rm(0,1)*rm(2,0)),                                 // col 0
            rm(0,1)*rm(2,2) + rm(0,2)*rm(2,1),                                    // col 1
            (s3/3.0) * (rm(0,0)*rm(2,0) + rm(0,1)*rm(2,1) - 2.0*rm(0,2)*rm(2,2)), // col 2
            rm(0,0)*rm(2,2) + rm(0,2)*rm(2,0),                                    // col 3
            -(rm(0,0)*rm(2,0) - rm(0,1)*rm(2,1)),                                 // col 4
        ],
        // Row 4 (m' = 2, Y₂² ~ x²-y²)
        [
            rm(0,0)*rm(0,1) - rm(1,0)*rm(1,1),                                    // col 0
            -(rm(0,1)*rm(0,2) - rm(1,1)*rm(1,2)),                                 // col 1
            (s3/6.0) * (-(rm(0,0)*rm(0,0)) - rm(0,1)*rm(0,1) + rm(1,0)*rm(1,0) + rm(1,1)*rm(1,1) + 2.0*rm(0,2)*rm(0,2) - 2.0*rm(1,2)*rm(1,2)), // col 2
            -(rm(0,0)*rm(0,2) - rm(1,0)*rm(1,2)),                                 // col 3
            0.5 * (rm(0,0)*rm(0,0) - rm(0,1)*rm(0,1) - rm(1,0)*rm(1,0) + rm(1,1)*rm(1,1)), // col 4
        ],
    ]
}

/// Compute the 7×7 rotation matrix for degree-3 (l=3) spherical harmonics.
///
/// Uses the Ivanic-Ruedenberg recurrence to build l=3 from l=1 and l=2.
/// This follows Google's spherical-harmonics library implementation.
pub fn sh_rotation_matrix_l3(rot: Mat3) -> [[f32; 7]; 7] {
    let r1 = sh_rotation_matrix_l1(rot);
    let r2 = sh_rotation_matrix_l2(rot);
    
    compute_band_rotation_l3(&r1, &r2)
}

/// Compute the 9×9 rotation matrix for degree-4 (l=4) spherical harmonics.
///
/// Uses the Ivanic-Ruedenberg recurrence to build l=4 from l=1 and l=3.
pub fn sh_rotation_matrix_l4(rot: Mat3) -> [[f32; 9]; 9] {
    let r1 = sh_rotation_matrix_l1(rot);
    let r3 = sh_rotation_matrix_l3(rot);
    
    compute_band_rotation_l4(&r1, &r3)
}

/// Compute the rotation matrix for band l=3 using the Ivanic-Ruedenberg recurrence.
fn compute_band_rotation_l3(
    r1: &[[f32; 3]; 3],
    r_prev: &[[f32; 5]; 5],
) -> [[f32; 7]; 7] {
    let l = 3;
    let l_prev = 2;
    let mut result = [[0.0f32; 7]; 7];
    
    for m in -l..=l {
        for n in -l..=l {
            let (u, v, w) = compute_uvw_coeff(m, n, l);
            
            let mut val = 0.0;
            
            if u.abs() > 1e-10 {
                val += u * compute_p_generic(0, m, n, l, l_prev, r1, r_prev);
            }
            if v.abs() > 1e-10 {
                val += v * compute_v_generic(m, n, l, l_prev, r1, r_prev);
            }
            if w.abs() > 1e-10 {
                val += w * compute_w_generic(m, n, l, l_prev, r1, r_prev);
            }
            
            result[(m + l) as usize][(n + l) as usize] = val;
        }
    }
    
    result
}

/// Compute the rotation matrix for band l=4 using the Ivanic-Ruedenberg recurrence.
fn compute_band_rotation_l4(
    r1: &[[f32; 3]; 3],
    r_prev: &[[f32; 7]; 7],
) -> [[f32; 9]; 9] {
    let l = 4;
    let l_prev = 3;
    let mut result = [[0.0f32; 9]; 9];
    
    for m in -l..=l {
        for n in -l..=l {
            let (u, v, w) = compute_uvw_coeff(m, n, l);
            
            let mut val = 0.0;
            
            if u.abs() > 1e-10 {
                val += u * compute_p_generic_l4(0, m, n, l, l_prev, r1, r_prev);
            }
            if v.abs() > 1e-10 {
                val += v * compute_v_generic_l4(m, n, l, l_prev, r1, r_prev);
            }
            if w.abs() > 1e-10 {
                val += w * compute_w_generic_l4(m, n, l, l_prev, r1, r_prev);
            }
            
            result[(m + l) as usize][(n + l) as usize] = val;
        }
    }
    
    result
}

/// Get element from centered indices (Google's convention).
/// For a (2l+1) x (2l+1) matrix, indices go from -l to l.
fn get_centered_l1(r1: &[[f32; 3]; 3], i: i32, j: i32) -> f32 {
    // l=1, so offset = 1
    r1[(i + 1) as usize][(j + 1) as usize]
}

fn get_centered_prev<const N: usize>(r_prev: &[[f32; N]; N], l_prev: i32, i: i32, j: i32) -> f32 {
    let row = i + l_prev;
    let col = j + l_prev;
    if row >= 0 && row < N as i32 && col >= 0 && col < N as i32 {
        r_prev[row as usize][col as usize]
    } else {
        0.0
    }
}

/// P function from Ivanic-Ruedenberg (generic version for l=3).
fn compute_p_generic(
    i: i32, a: i32, b: i32, l: i32, l_prev: i32,
    r1: &[[f32; 3]; 3], r_prev: &[[f32; 5]; 5]
) -> f32 {
    if b == l {
        get_centered_l1(r1, i, 1) * get_centered_prev(r_prev, l_prev, a, l - 1) -
        get_centered_l1(r1, i, -1) * get_centered_prev(r_prev, l_prev, a, -l + 1)
    } else if b == -l {
        get_centered_l1(r1, i, 1) * get_centered_prev(r_prev, l_prev, a, -l + 1) +
        get_centered_l1(r1, i, -1) * get_centered_prev(r_prev, l_prev, a, l - 1)
    } else {
        get_centered_l1(r1, i, 0) * get_centered_prev(r_prev, l_prev, a, b)
    }
}

/// P function from Ivanic-Ruedenberg (for l=4, using l=3 as previous).
fn compute_p_generic_l4(
    i: i32, a: i32, b: i32, l: i32, l_prev: i32,
    r1: &[[f32; 3]; 3], r_prev: &[[f32; 7]; 7]
) -> f32 {
    if b == l {
        get_centered_l1(r1, i, 1) * get_centered_prev(r_prev, l_prev, a, l - 1) -
        get_centered_l1(r1, i, -1) * get_centered_prev(r_prev, l_prev, a, -l + 1)
    } else if b == -l {
        get_centered_l1(r1, i, 1) * get_centered_prev(r_prev, l_prev, a, -l + 1) +
        get_centered_l1(r1, i, -1) * get_centered_prev(r_prev, l_prev, a, l - 1)
    } else {
        get_centered_l1(r1, i, 0) * get_centered_prev(r_prev, l_prev, a, b)
    }
}

/// V function from Ivanic-Ruedenberg (generic for l=3).
fn compute_v_generic(
    m: i32, n: i32, l: i32, l_prev: i32,
    r1: &[[f32; 3]; 3], r_prev: &[[f32; 5]; 5]
) -> f32 {
    if m == 0 {
        compute_p_generic(1, 1, n, l, l_prev, r1, r_prev) +
        compute_p_generic(-1, -1, n, l, l_prev, r1, r_prev)
    } else if m > 0 {
        let d = if m == 1 { 1.0_f32 } else { 0.0_f32 };
        compute_p_generic(1, m - 1, n, l, l_prev, r1, r_prev) * (1.0 + d).sqrt() -
        compute_p_generic(-1, -m + 1, n, l, l_prev, r1, r_prev) * (1.0 - d)
    } else {
        let d = if m == -1 { 1.0_f32 } else { 0.0_f32 };
        compute_p_generic(1, m + 1, n, l, l_prev, r1, r_prev) * (1.0 - d) +
        compute_p_generic(-1, -m - 1, n, l, l_prev, r1, r_prev) * (1.0 + d).sqrt()
    }
}

/// V function from Ivanic-Ruedenberg (for l=4).
fn compute_v_generic_l4(
    m: i32, n: i32, l: i32, l_prev: i32,
    r1: &[[f32; 3]; 3], r_prev: &[[f32; 7]; 7]
) -> f32 {
    if m == 0 {
        compute_p_generic_l4(1, 1, n, l, l_prev, r1, r_prev) +
        compute_p_generic_l4(-1, -1, n, l, l_prev, r1, r_prev)
    } else if m > 0 {
        let d = if m == 1 { 1.0_f32 } else { 0.0_f32 };
        compute_p_generic_l4(1, m - 1, n, l, l_prev, r1, r_prev) * (1.0 + d).sqrt() -
        compute_p_generic_l4(-1, -m + 1, n, l, l_prev, r1, r_prev) * (1.0 - d)
    } else {
        let d = if m == -1 { 1.0_f32 } else { 0.0_f32 };
        compute_p_generic_l4(1, m + 1, n, l, l_prev, r1, r_prev) * (1.0 - d) +
        compute_p_generic_l4(-1, -m - 1, n, l, l_prev, r1, r_prev) * (1.0 + d).sqrt()
    }
}

/// W function from Ivanic-Ruedenberg (generic for l=3).
fn compute_w_generic(
    m: i32, n: i32, l: i32, l_prev: i32,
    r1: &[[f32; 3]; 3], r_prev: &[[f32; 5]; 5]
) -> f32 {
    if m == 0 {
        0.0
    } else if m > 0 {
        compute_p_generic(1, m + 1, n, l, l_prev, r1, r_prev) +
        compute_p_generic(-1, -m - 1, n, l, l_prev, r1, r_prev)
    } else {
        compute_p_generic(1, m - 1, n, l, l_prev, r1, r_prev) -
        compute_p_generic(-1, -m + 1, n, l, l_prev, r1, r_prev)
    }
}

/// W function from Ivanic-Ruedenberg (for l=4).
fn compute_w_generic_l4(
    m: i32, n: i32, l: i32, l_prev: i32,
    r1: &[[f32; 3]; 3], r_prev: &[[f32; 7]; 7]
) -> f32 {
    if m == 0 {
        0.0
    } else if m > 0 {
        compute_p_generic_l4(1, m + 1, n, l, l_prev, r1, r_prev) +
        compute_p_generic_l4(-1, -m - 1, n, l, l_prev, r1, r_prev)
    } else {
        compute_p_generic_l4(1, m - 1, n, l, l_prev, r1, r_prev) -
        compute_p_generic_l4(-1, -m + 1, n, l, l_prev, r1, r_prev)
    }
}

/// Compute the UVW coefficients for the recurrence.
fn compute_uvw_coeff(m: i32, n: i32, l: i32) -> (f32, f32, f32) {
    let d = if m == 0 { 1.0_f32 } else { 0.0_f32 };
    let abs_m = m.abs() as f32;
    let abs_n = n.abs() as f32;
    let lf = l as f32;
    let mf = m as f32;
    let nf = n as f32;
    
    let denom = if abs_n == lf {
        2.0 * lf * (2.0 * lf - 1.0)
    } else {
        (lf + nf) * (lf - nf)
    };
    
    let u = ((lf + mf) * (lf - mf) / denom).sqrt();
    let v = 0.5 * ((1.0 + d) * (lf + abs_m - 1.0) * (lf + abs_m) / denom).sqrt() * (1.0 - 2.0 * d);
    let w = -0.5 * ((lf - abs_m - 1.0) * (lf - abs_m) / denom).sqrt() * (1.0 - d);
    
    (u, v, w)
}

/// Precomputed rotation matrices for all supported SH degrees.
pub struct ShRotationMatrices {
    /// 3×3 rotation matrix for l=1
    pub l1: [[f32; 3]; 3],
    /// 5×5 rotation matrix for l=2
    pub l2: [[f32; 5]; 5],
    /// 7×7 rotation matrix for l=3
    pub l3: [[f32; 7]; 7],
    /// 9×9 rotation matrix for l=4
    pub l4: [[f32; 9]; 9],
}

impl ShRotationMatrices {
    /// Compute rotation matrices for all SH degrees from a quaternion.
    pub fn from_quat(quat: Quat) -> Self {
        let rot = Mat3::from_quat(quat);
        Self::from_mat3(rot)
    }
    
    /// Compute rotation matrices for all SH degrees from a 3×3 rotation matrix.
    pub fn from_mat3(rot: Mat3) -> Self {
        Self {
            l1: sh_rotation_matrix_l1(rot),
            l2: sh_rotation_matrix_l2(rot),
            l3: sh_rotation_matrix_l3(rot),
            l4: sh_rotation_matrix_l4(rot),
        }
    }
}

/// Rotate spherical harmonic coefficients in-place.
///
/// # Arguments
/// * `sh_coeffs` - Flattened SH coefficients in the layout used after permute:
///                 `[N, 3, coeffs_per_channel]` flattened, where 3 is RGB channels.
///                 Each Gaussian has `3 * coeffs_per_channel` consecutive floats.
/// * `sh_degree` - The SH degree (0-3). Degree 0 has 1 coeff, degree 1 has 4, etc.
/// * `rotation` - The rotation quaternion to apply.
///
/// Note: Degree 0 (DC term) is rotationally invariant and not modified.
pub fn rotate_sh_coefficients_in_place(
    sh_coeffs: &mut [f32],
    sh_degree: u32,
    rotation: Quat,
) {
    if sh_degree == 0 {
        return;
    }
    
    let matrices = ShRotationMatrices::from_quat(rotation);
    let coeffs_per_channel = ((sh_degree + 1) * (sh_degree + 1)) as usize;
    let floats_per_gaussian = 3 * coeffs_per_channel;
    
    for gaussian_coeffs in sh_coeffs.chunks_exact_mut(floats_per_gaussian) {
        for channel in 0..3 {
            let channel_offset = channel * coeffs_per_channel;
            let channel_slice = &mut gaussian_coeffs[channel_offset..channel_offset + coeffs_per_channel];
            
            if sh_degree >= 1 {
                rotate_band_in_place(&mut channel_slice[1..4], &matrices.l1);
            }
            
            if sh_degree >= 2 {
                rotate_band_in_place(&mut channel_slice[4..9], &matrices.l2);
            }
            
            if sh_degree >= 3 {
                rotate_band_in_place(&mut channel_slice[9..16], &matrices.l3);
            }
            
            if sh_degree >= 4 {
                rotate_band_in_place(&mut channel_slice[16..25], &matrices.l4);
            }
        }
    }
}

/// Apply a rotation matrix to a single SH band in-place.
fn rotate_band_in_place<const N: usize>(coeffs: &mut [f32], matrix: &[[f32; N]; N]) {
    debug_assert_eq!(coeffs.len(), N);
    
    let mut input = [0.0f32; N];
    input[..N].copy_from_slice(&coeffs[..N]);
    
    for i in 0..N {
        let mut sum = 0.0;
        for j in 0..N {
            sum += matrix[i][j] * input[j];
        }
        coeffs[i] = sum;
    }
}

/// Rotate spherical harmonic coefficients in-place for the interleaved layout.
///
/// This is for the layout `[N, coeffs_per_channel, 3]` where each SH coefficient
/// has R, G, B values interleaved (used by SPZ format and Brush's native layout).
pub fn rotate_sh_coefficients_interleaved_in_place(
    sh_coeffs: &mut [f32],
    sh_degree: u32,
    rotation: Quat,
) {
    if sh_degree == 0 {
        return;
    }
    
    let matrices = ShRotationMatrices::from_quat(rotation);
    let coeffs_per_channel = ((sh_degree + 1) * (sh_degree + 1)) as usize;
    let floats_per_gaussian = coeffs_per_channel * 3;
    
    let mut band_r = [0.0f32; 9];
    let mut band_g = [0.0f32; 9];
    let mut band_b = [0.0f32; 9];
    
    for gaussian_coeffs in sh_coeffs.chunks_exact_mut(floats_per_gaussian) {
        if sh_degree >= 1 {
            for (idx, coeff_idx) in (1..=3).enumerate() {
                let base = coeff_idx * 3;
                band_r[idx] = gaussian_coeffs[base + 0];
                band_g[idx] = gaussian_coeffs[base + 1];
                band_b[idx] = gaussian_coeffs[base + 2];
            }
            
            rotate_band_array::<3>(&mut band_r, &matrices.l1);
            rotate_band_array::<3>(&mut band_g, &matrices.l1);
            rotate_band_array::<3>(&mut band_b, &matrices.l1);
            
            for (idx, coeff_idx) in (1..=3).enumerate() {
                let base = coeff_idx * 3;
                gaussian_coeffs[base + 0] = band_r[idx];
                gaussian_coeffs[base + 1] = band_g[idx];
                gaussian_coeffs[base + 2] = band_b[idx];
            }
        }
        
        if sh_degree >= 2 {
            for (idx, coeff_idx) in (4..=8).enumerate() {
                let base = coeff_idx * 3;
                band_r[idx] = gaussian_coeffs[base + 0];
                band_g[idx] = gaussian_coeffs[base + 1];
                band_b[idx] = gaussian_coeffs[base + 2];
            }
            
            rotate_band_array::<5>(&mut band_r, &matrices.l2);
            rotate_band_array::<5>(&mut band_g, &matrices.l2);
            rotate_band_array::<5>(&mut band_b, &matrices.l2);
            
            for (idx, coeff_idx) in (4..=8).enumerate() {
                let base = coeff_idx * 3;
                gaussian_coeffs[base + 0] = band_r[idx];
                gaussian_coeffs[base + 1] = band_g[idx];
                gaussian_coeffs[base + 2] = band_b[idx];
            }
        }
        
        if sh_degree >= 3 {
            for (idx, coeff_idx) in (9..=15).enumerate() {
                let base = coeff_idx * 3;
                band_r[idx] = gaussian_coeffs[base + 0];
                band_g[idx] = gaussian_coeffs[base + 1];
                band_b[idx] = gaussian_coeffs[base + 2];
            }
            
            rotate_band_array::<7>(&mut band_r, &matrices.l3);
            rotate_band_array::<7>(&mut band_g, &matrices.l3);
            rotate_band_array::<7>(&mut band_b, &matrices.l3);
            
            for (idx, coeff_idx) in (9..=15).enumerate() {
                let base = coeff_idx * 3;
                gaussian_coeffs[base + 0] = band_r[idx];
                gaussian_coeffs[base + 1] = band_g[idx];
                gaussian_coeffs[base + 2] = band_b[idx];
            }
        }
        
        if sh_degree >= 4 {
            for (idx, coeff_idx) in (16..=24).enumerate() {
                let base = coeff_idx * 3;
                band_r[idx] = gaussian_coeffs[base + 0];
                band_g[idx] = gaussian_coeffs[base + 1];
                band_b[idx] = gaussian_coeffs[base + 2];
            }
            
            rotate_band_array::<9>(&mut band_r, &matrices.l4);
            rotate_band_array::<9>(&mut band_g, &matrices.l4);
            rotate_band_array::<9>(&mut band_b, &matrices.l4);
            
            for (idx, coeff_idx) in (16..=24).enumerate() {
                let base = coeff_idx * 3;
                gaussian_coeffs[base + 0] = band_r[idx];
                gaussian_coeffs[base + 1] = band_g[idx];
                gaussian_coeffs[base + 2] = band_b[idx];
            }
        }
    }
}

/// Apply a rotation matrix to a band stored in a larger buffer.
fn rotate_band_array<const N: usize>(coeffs: &mut [f32; 9], matrix: &[[f32; N]; N]) {
    let mut input = [0.0f32; N];
    input[..N].copy_from_slice(&coeffs[..N]);
    
    for i in 0..N {
        let mut sum = 0.0;
        for j in 0..N {
            sum += matrix[i][j] * input[j];
        }
        coeffs[i] = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f32::consts::{FRAC_PI_2, PI};
    
    // ========================================================================
    // Identity rotation tests
    // ========================================================================
    
    #[test]
    fn test_identity_rotation_l1() {
        let identity = Quat::IDENTITY;
        let m1 = sh_rotation_matrix_l1(Mat3::from_quat(identity));
        
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(m1[i][j], expected, epsilon = 1e-5);
            }
        }
    }
    
    #[test]
    fn test_identity_rotation_l2() {
        let identity = Quat::IDENTITY;
        let m2 = sh_rotation_matrix_l2(Mat3::from_quat(identity));
        
        for i in 0..5 {
            for j in 0..5 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(m2[i][j], expected, epsilon = 1e-5);
            }
        }
    }
    
    #[test]
    fn test_identity_rotation_l3() {
        let identity = Quat::IDENTITY;
        let m3 = sh_rotation_matrix_l3(Mat3::from_quat(identity));
        
        for i in 0..7 {
            for j in 0..7 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(m3[i][j], expected, epsilon = 1e-4);
            }
        }
    }
    
    #[test]
    fn test_identity_rotation_l4() {
        let identity = Quat::IDENTITY;
        let m4 = sh_rotation_matrix_l4(Mat3::from_quat(identity));
        
        for i in 0..9 {
            for j in 0..9 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(m4[i][j], expected, epsilon = 1e-4);
            }
        }
    }
    
    // ========================================================================
    // Orthonormality tests (rotation matrices should be orthogonal)
    // ========================================================================
    
    fn check_orthonormality<const N: usize>(matrix: &[[f32; N]; N], epsilon: f32) {
        // Check M * M^T = I
        for i in 0..N {
            for j in 0..N {
                let mut dot = 0.0;
                for k in 0..N {
                    dot += matrix[i][k] * matrix[j][k];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < epsilon, 
                    "row {} · row {} = {} (expected {})", i, j, dot, expected);
            }
        }
    }
    
    #[test]
    fn test_orthonormality_l1() {
        let quat = Quat::from_euler(glam::EulerRot::XYZ, 0.3, 0.5, 0.7);
        let m1 = sh_rotation_matrix_l1(Mat3::from_quat(quat));
        check_orthonormality(&m1, 1e-4);
    }
    
    #[test]
    fn test_orthonormality_l2() {
        let quat = Quat::from_euler(glam::EulerRot::XYZ, 0.3, 0.5, 0.7);
        let m2 = sh_rotation_matrix_l2(Mat3::from_quat(quat));
        check_orthonormality(&m2, 1e-4);
    }
    
    #[test]
    fn test_orthonormality_l3() {
        let quat = Quat::from_euler(glam::EulerRot::XYZ, 0.3, 0.5, 0.7);
        let m3 = sh_rotation_matrix_l3(Mat3::from_quat(quat));
        check_orthonormality(&m3, 1e-3);
    }
    
    #[test]
    fn test_orthonormality_l4() {
        let quat = Quat::from_euler(glam::EulerRot::XYZ, 0.3, 0.5, 0.7);
        let m4 = sh_rotation_matrix_l4(Mat3::from_quat(quat));
        check_orthonormality(&m4, 1e-3);
    }
    
    // ========================================================================
    // Axis-aligned rotation tests with hard-coded reference values
    // ========================================================================
    
    #[test]
    fn test_180_rotation_z() {
        let quat = Quat::from_rotation_z(PI);
        let m1 = sh_rotation_matrix_l1(Mat3::from_quat(quat));
        
        // After 180° about Z: x -> -x, y -> -y, z -> z
        assert_relative_eq!(m1[0][0], -1.0, epsilon = 1e-5);
        assert_relative_eq!(m1[1][1], 1.0, epsilon = 1e-5);
        assert_relative_eq!(m1[2][2], -1.0, epsilon = 1e-5);
    }
    
    #[test]
    fn test_90_rotation_z_l1() {
        // 90° rotation about Z: x -> y, y -> -x, z -> z
        let quat = Quat::from_rotation_z(FRAC_PI_2);
        let m1 = sh_rotation_matrix_l1(Mat3::from_quat(quat));
        
        // The z component (Y_{1,0}) should be unchanged
        assert_relative_eq!(m1[1][1], 1.0, epsilon = 1e-5);
        
        // The x and y components should mix (diagonals should be zero)
        assert_relative_eq!(m1[0][0].abs() + m1[2][2].abs(), 0.0, epsilon = 1e-5);
    }
    
    #[test]
    fn test_90_rotation_x_l1() {
        // 90° rotation about X: x -> x, y -> z, z -> -y
        let quat = Quat::from_rotation_x(FRAC_PI_2);
        let m1 = sh_rotation_matrix_l1(Mat3::from_quat(quat));
        
        // The x component should be unchanged (appears in index 2 in our basis)
        assert_relative_eq!(m1[2][2], 1.0, epsilon = 1e-5);
    }
    
    // ========================================================================
    // Round-trip tests (rotate then inverse-rotate should give original)
    // ========================================================================
    
    #[test]
    fn test_round_trip_l1() {
        let mut coeffs: Vec<f32> = vec![
            1.0, 0.1, 0.2, 0.3,  // R: DC, l1 (3 coeffs)
            0.5, 0.4, 0.5, 0.6,  // G
            0.3, 0.7, 0.8, 0.9,  // B
        ];
        
        let original = coeffs.clone();
        let quat = Quat::from_euler(glam::EulerRot::XYZ, 0.3, 0.5, 0.7);
        let quat_inv = quat.inverse();
        
        rotate_sh_coefficients_in_place(&mut coeffs, 1, quat);
        rotate_sh_coefficients_in_place(&mut coeffs, 1, quat_inv);
        
        for (i, (orig, result)) in original.iter().zip(coeffs.iter()).enumerate() {
            assert!((result - orig).abs() < 1e-4, 
                "L1 round-trip failed at index {}: {} vs {}", i, result, orig);
        }
    }
    
    #[test]
    fn test_round_trip_l2() {
        // L2 has 9 coefficients per channel (1 DC + 3 L1 + 5 L2)
        let mut coeffs: Vec<f32> = vec![
            // R channel
            1.0,                         // DC
            0.1, 0.2, 0.3,               // L1
            0.11, 0.12, 0.13, 0.14, 0.15, // L2
            // G channel
            0.5,
            0.4, 0.5, 0.6,
            0.21, 0.22, 0.23, 0.24, 0.25,
            // B channel
            0.3,
            0.7, 0.8, 0.9,
            0.31, 0.32, 0.33, 0.34, 0.35,
        ];
        
        let original = coeffs.clone();
        let quat = Quat::from_euler(glam::EulerRot::XYZ, 0.3, 0.5, 0.7);
        let quat_inv = quat.inverse();
        
        rotate_sh_coefficients_in_place(&mut coeffs, 2, quat);
        rotate_sh_coefficients_in_place(&mut coeffs, 2, quat_inv);
        
        for (i, (orig, result)) in original.iter().zip(coeffs.iter()).enumerate() {
            assert!((result - orig).abs() < 1e-4, 
                "L2 round-trip failed at index {}: {} vs {}", i, result, orig);
        }
    }
    
    #[test]
    fn test_round_trip_l3() {
        // L3 has 16 coefficients per channel (1 + 3 + 5 + 7)
        let mut coeffs: Vec<f32> = vec![
            // R channel
            1.0,                                    // DC
            0.1, 0.2, 0.3,                          // L1
            0.11, 0.12, 0.13, 0.14, 0.15,           // L2
            0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, // L3
            // G channel
            0.5,
            0.4, 0.5, 0.6,
            0.41, 0.42, 0.43, 0.44, 0.45,
            0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57,
            // B channel
            0.3,
            0.7, 0.8, 0.9,
            0.61, 0.62, 0.63, 0.64, 0.65,
            0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77,
        ];
        
        let original = coeffs.clone();
        let quat = Quat::from_euler(glam::EulerRot::XYZ, 0.3, 0.5, 0.7);
        let quat_inv = quat.inverse();
        
        rotate_sh_coefficients_in_place(&mut coeffs, 3, quat);
        rotate_sh_coefficients_in_place(&mut coeffs, 3, quat_inv);
        
        for (i, (orig, result)) in original.iter().zip(coeffs.iter()).enumerate() {
            assert!((result - orig).abs() < 1e-3, 
                "L3 round-trip failed at index {}: {} vs {}", i, result, orig);
        }
    }
    
    #[test]
    fn test_round_trip_l4() {
        // L4 has 25 coefficients per channel (1 + 3 + 5 + 7 + 9)
        let mut coeffs: Vec<f32> = vec![
            // R channel
            1.0,                                            // DC
            0.1, 0.2, 0.3,                                  // L1
            0.11, 0.12, 0.13, 0.14, 0.15,                   // L2
            0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27,       // L3
            0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, // L4
            // G channel
            0.5,
            0.4, 0.5, 0.6,
            0.41, 0.42, 0.43, 0.44, 0.45,
            0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57,
            0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.69,
            // B channel
            0.3,
            0.7, 0.8, 0.9,
            0.61, 0.62, 0.63, 0.64, 0.65,
            0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77,
            0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89,
        ];
        
        let original = coeffs.clone();
        let quat = Quat::from_euler(glam::EulerRot::XYZ, 0.3, 0.5, 0.7);
        let quat_inv = quat.inverse();
        
        rotate_sh_coefficients_in_place(&mut coeffs, 4, quat);
        rotate_sh_coefficients_in_place(&mut coeffs, 4, quat_inv);
        
        for (i, (orig, result)) in original.iter().zip(coeffs.iter()).enumerate() {
            assert!((result - orig).abs() < 1e-3, 
                "L4 round-trip failed at index {}: {} vs {}", i, result, orig);
        }
    }
    
    #[test]
    fn test_interleaved_round_trip_l1() {
        let mut coeffs: Vec<f32> = vec![
            1.0, 0.5, 0.3,  // DC: RGB
            0.1, 0.4, 0.7,  // L1_0: RGB
            0.2, 0.5, 0.8,  // L1_1: RGB
            0.3, 0.6, 0.9,  // L1_2: RGB
        ];
        
        let original = coeffs.clone();
        let quat = Quat::from_euler(glam::EulerRot::XYZ, 0.3, 0.5, 0.7);
        let quat_inv = quat.inverse();
        
        rotate_sh_coefficients_interleaved_in_place(&mut coeffs, 1, quat);
        rotate_sh_coefficients_interleaved_in_place(&mut coeffs, 1, quat_inv);
        
        for (i, (orig, result)) in original.iter().zip(coeffs.iter()).enumerate() {
            assert!((result - orig).abs() < 1e-4, 
                "Interleaved L1 round-trip failed at index {}: {} vs {}", i, result, orig);
        }
    }
    
    #[test]
    fn test_interleaved_round_trip_l3() {
        // L3 has 16 coefficients, each with RGB = 48 floats total
        let mut coeffs: Vec<f32> = (0..48).map(|i| i as f32 * 0.1).collect();
        
        let original = coeffs.clone();
        let quat = Quat::from_euler(glam::EulerRot::XYZ, 0.3, 0.5, 0.7);
        let quat_inv = quat.inverse();
        
        rotate_sh_coefficients_interleaved_in_place(&mut coeffs, 3, quat);
        rotate_sh_coefficients_interleaved_in_place(&mut coeffs, 3, quat_inv);
        
        for (i, (orig, result)) in original.iter().zip(coeffs.iter()).enumerate() {
            assert!((result - orig).abs() < 1e-3, 
                "Interleaved L3 round-trip failed at index {}: {} vs {}", i, result, orig);
        }
    }
    
    #[test]
    fn test_interleaved_round_trip_l4() {
        // L4 has 25 coefficients, each with RGB = 75 floats total
        let mut coeffs: Vec<f32> = (0..75).map(|i| i as f32 * 0.1).collect();
        
        let original = coeffs.clone();
        let quat = Quat::from_euler(glam::EulerRot::XYZ, 0.3, 0.5, 0.7);
        let quat_inv = quat.inverse();
        
        rotate_sh_coefficients_interleaved_in_place(&mut coeffs, 4, quat);
        rotate_sh_coefficients_interleaved_in_place(&mut coeffs, 4, quat_inv);
        
        for (i, (orig, result)) in original.iter().zip(coeffs.iter()).enumerate() {
            assert!((result - orig).abs() < 1e-3, 
                "Interleaved L4 round-trip failed at index {}: {} vs {}", i, result, orig);
        }
    }
    
    // ========================================================================
    // Energy conservation tests (rotation should preserve L2 norm)
    // ========================================================================
    
    fn l2_norm(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
    
    #[test]
    fn test_energy_conservation_l1() {
        let band = [0.1f32, 0.2, 0.3];
        let original_energy = l2_norm(&band);
        
        let quat = Quat::from_euler(glam::EulerRot::XYZ, 0.3, 0.5, 0.7);
        let m1 = sh_rotation_matrix_l1(Mat3::from_quat(quat));
        
        let mut rotated = [0.0f32; 3];
        for i in 0..3 {
            for j in 0..3 {
                rotated[i] += m1[i][j] * band[j];
            }
        }
        
        let rotated_energy = l2_norm(&rotated);
        assert!((rotated_energy - original_energy).abs() < 1e-5,
            "L1 energy not conserved: {} vs {}", rotated_energy, original_energy);
    }
    
    #[test]
    fn test_energy_conservation_l2() {
        let band = [0.1f32, 0.2, 0.3, 0.4, 0.5];
        let original_energy = l2_norm(&band);
        
        let quat = Quat::from_euler(glam::EulerRot::XYZ, 0.3, 0.5, 0.7);
        let m2 = sh_rotation_matrix_l2(Mat3::from_quat(quat));
        
        let mut rotated = [0.0f32; 5];
        for i in 0..5 {
            for j in 0..5 {
                rotated[i] += m2[i][j] * band[j];
            }
        }
        
        let rotated_energy = l2_norm(&rotated);
        assert!((rotated_energy - original_energy).abs() < 1e-4,
            "L2 energy not conserved: {} vs {}", rotated_energy, original_energy);
    }
    
    #[test]
    fn test_energy_conservation_l3() {
        let band = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7];
        let original_energy = l2_norm(&band);
        
        let quat = Quat::from_euler(glam::EulerRot::XYZ, 0.3, 0.5, 0.7);
        let m3 = sh_rotation_matrix_l3(Mat3::from_quat(quat));
        
        let mut rotated = [0.0f32; 7];
        for i in 0..7 {
            for j in 0..7 {
                rotated[i] += m3[i][j] * band[j];
            }
        }
        
        let rotated_energy = l2_norm(&rotated);
        assert!((rotated_energy - original_energy).abs() < 1e-3,
            "L3 energy not conserved: {} vs {}", rotated_energy, original_energy);
    }
    
    #[test]
    fn test_energy_conservation_l4() {
        let band = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let original_energy = l2_norm(&band);
        
        let quat = Quat::from_euler(glam::EulerRot::XYZ, 0.3, 0.5, 0.7);
        let m4 = sh_rotation_matrix_l4(Mat3::from_quat(quat));
        
        let mut rotated = [0.0f32; 9];
        for i in 0..9 {
            for j in 0..9 {
                rotated[i] += m4[i][j] * band[j];
            }
        }
        
        let rotated_energy = l2_norm(&rotated);
        assert!((rotated_energy - original_energy).abs() < 1e-3,
            "L4 energy not conserved: {} vs {}", rotated_energy, original_energy);
    }
    
    // ========================================================================
    // DC term invariance test
    // ========================================================================
    
    #[test]
    fn test_dc_term_unchanged() {
        let dc_value = 0.5644; // √(1/4π) ≈ SH normalization constant
        
        let mut coeffs: Vec<f32> = vec![
            dc_value, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,  // R
            dc_value, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,  // G
            dc_value, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,  // B
        ];
        
        let quat = Quat::from_euler(glam::EulerRot::XYZ, 0.3, 0.5, 0.7);
        rotate_sh_coefficients_in_place(&mut coeffs, 2, quat);
        
        // DC terms should be unchanged
        assert_relative_eq!(coeffs[0], dc_value, epsilon = 1e-6);
        assert_relative_eq!(coeffs[9], dc_value, epsilon = 1e-6);
        assert_relative_eq!(coeffs[18], dc_value, epsilon = 1e-6);
    }
    
    // ========================================================================
    // Multiple Gaussian tests
    // ========================================================================
    
    #[test]
    fn test_multiple_gaussians_l2() {
        // Two Gaussians, L2 (9 coeffs per channel, 27 per Gaussian)
        let mut coeffs: Vec<f32> = (0..54).map(|i| i as f32 * 0.01).collect();
        let original = coeffs.clone();
        
        let quat = Quat::from_euler(glam::EulerRot::XYZ, 0.3, 0.5, 0.7);
        let quat_inv = quat.inverse();
        
        rotate_sh_coefficients_in_place(&mut coeffs, 2, quat);
        rotate_sh_coefficients_in_place(&mut coeffs, 2, quat_inv);
        
        for (i, (orig, result)) in original.iter().zip(coeffs.iter()).enumerate() {
            assert!((result - orig).abs() < 1e-4,
                "Multiple Gaussians L2 round-trip failed at index {}: {} vs {}", i, result, orig);
        }
    }
    
    // ========================================================================
    // Composition test (R1 * R2 should equal R(R1*R2))
    // ========================================================================
    
    #[test]
    fn test_rotation_composition_l1() {
        let quat1 = Quat::from_rotation_x(0.5);
        let quat2 = Quat::from_rotation_y(0.7);
        let quat_combined = quat2 * quat1; // Apply quat1 then quat2
        
        let m1_a = sh_rotation_matrix_l1(Mat3::from_quat(quat1));
        let m1_b = sh_rotation_matrix_l1(Mat3::from_quat(quat2));
        let m1_combined = sh_rotation_matrix_l1(Mat3::from_quat(quat_combined));
        
        // Compute m1_b * m1_a
        let mut composed = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    composed[i][j] += m1_b[i][k] * m1_a[k][j];
                }
            }
        }
        
        for i in 0..3 {
            for j in 0..3 {
                assert!((composed[i][j] - m1_combined[i][j]).abs() < 1e-4,
                    "Composition failed at [{}, {}]: {} vs {}", i, j, composed[i][j], m1_combined[i][j]);
            }
        }
    }
    
    #[test]
    fn test_rotation_composition_l2() {
        let quat1 = Quat::from_rotation_x(0.5);
        let quat2 = Quat::from_rotation_y(0.7);
        let quat_combined = quat2 * quat1;
        
        let m2_a = sh_rotation_matrix_l2(Mat3::from_quat(quat1));
        let m2_b = sh_rotation_matrix_l2(Mat3::from_quat(quat2));
        let m2_combined = sh_rotation_matrix_l2(Mat3::from_quat(quat_combined));
        
        let mut composed = [[0.0f32; 5]; 5];
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    composed[i][j] += m2_b[i][k] * m2_a[k][j];
                }
            }
        }
        
        for i in 0..5 {
            for j in 0..5 {
                assert!((composed[i][j] - m2_combined[i][j]).abs() < 1e-4,
                    "L2 composition failed at [{}, {}]: {} vs {}", i, j, composed[i][j], m2_combined[i][j]);
            }
        }
    }
    
    #[test]
    fn test_rotation_composition_l3() {
        let quat1 = Quat::from_rotation_x(0.5);
        let quat2 = Quat::from_rotation_y(0.7);
        let quat_combined = quat2 * quat1;
        
        let m3_a = sh_rotation_matrix_l3(Mat3::from_quat(quat1));
        let m3_b = sh_rotation_matrix_l3(Mat3::from_quat(quat2));
        let m3_combined = sh_rotation_matrix_l3(Mat3::from_quat(quat_combined));
        
        let mut composed = [[0.0f32; 7]; 7];
        for i in 0..7 {
            for j in 0..7 {
                for k in 0..7 {
                    composed[i][j] += m3_b[i][k] * m3_a[k][j];
                }
            }
        }
        
        for i in 0..7 {
            for j in 0..7 {
                assert!((composed[i][j] - m3_combined[i][j]).abs() < 1e-3,
                    "L3 composition failed at [{}, {}]: {} vs {}", i, j, composed[i][j], m3_combined[i][j]);
            }
        }
    }
    
    #[test]
    fn test_rotation_composition_l4() {
        let quat1 = Quat::from_rotation_x(0.5);
        let quat2 = Quat::from_rotation_y(0.7);
        let quat_combined = quat2 * quat1;
        
        let m4_a = sh_rotation_matrix_l4(Mat3::from_quat(quat1));
        let m4_b = sh_rotation_matrix_l4(Mat3::from_quat(quat2));
        let m4_combined = sh_rotation_matrix_l4(Mat3::from_quat(quat_combined));
        
        let mut composed = [[0.0f32; 9]; 9];
        for i in 0..9 {
            for j in 0..9 {
                for k in 0..9 {
                    composed[i][j] += m4_b[i][k] * m4_a[k][j];
                }
            }
        }
        
        for i in 0..9 {
            for j in 0..9 {
                assert!((composed[i][j] - m4_combined[i][j]).abs() < 1e-3,
                    "L4 composition failed at [{}, {}]: {} vs {}", i, j, composed[i][j], m4_combined[i][j]);
            }
        }
    }
}
