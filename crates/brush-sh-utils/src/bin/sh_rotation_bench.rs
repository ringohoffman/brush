//! Benchmark for SH rotation functions
//! 
//! Run with: cargo bench -p brush-sh-utils

use brush_sh_utils::rotate_sh_coefficients_interleaved_in_place;
use glam::Quat;
use std::time::Instant;

fn main() {
    // Test various sizes representative of real scenes
    let sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000];
    let degrees = [1, 2, 3, 4];
    
    let rotation = Quat::from_euler(glam::EulerRot::XYZ, 0.3, 0.5, 0.7);
    
    println!("SH Rotation Benchmark");
    println!("=====================");
    println!();
    
    for &degree in &degrees {
        let coeffs_per_channel = ((degree + 1) * (degree + 1)) as usize;
        println!("SH Degree {}: {} coefficients per channel", degree, coeffs_per_channel);
        println!("-----------------------------------------");
        
        for &num_gaussians in &sizes {
            let floats_per_gaussian = coeffs_per_channel * 3;
            let mut data: Vec<f32> = (0..num_gaussians * floats_per_gaussian)
                .map(|i| (i as f32) * 0.001)
                .collect();
            
            // Warm up
            rotate_sh_coefficients_interleaved_in_place(&mut data, degree, rotation);
            
            // Time multiple iterations
            let iterations = 10;
            let start = Instant::now();
            for _ in 0..iterations {
                rotate_sh_coefficients_interleaved_in_place(&mut data, degree, rotation);
            }
            let elapsed = start.elapsed();
            let per_iter = elapsed / iterations;
            let throughput = num_gaussians as f64 / per_iter.as_secs_f64() / 1_000_000.0;
            
            println!(
                "  {:>8} gaussians: {:>8.2?} ({:.2} M gaussians/sec)",
                num_gaussians, per_iter, throughput
            );
        }
        println!();
    }
}
