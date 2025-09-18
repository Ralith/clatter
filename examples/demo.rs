#![feature(portable_simd)]

use std::{
    simd::{num::SimdFloat, LaneCount, Simd, SupportedLaneCount},
    time::Instant,
};

use clap::Parser;

use clatter::{Sample, Simplex2d};

/// Compute a patch of fractal brownian motion noise
#[derive(Parser)]
#[clap(name = "demo")]
struct Opts {
    /// Length of the sampled area's edges
    #[clap(short, default_value = "5.0")]
    scale: f32,
    /// Resolution of the output image
    #[clap(short, default_value = "512")]
    resolution: usize,
    /// Number of times to sample the noise
    #[clap(short, default_value = "4")]
    octaves: usize,
    /// How smooth the noise should be (sensible values are around 0.5-1)
    #[clap(short = 'e', default_value = "1.0")]
    hurst_exponent: f32,
    /// Frequency ratio between successive octaves
    #[clap(short, default_value = "1.9")]
    lacunarity: f32,
}

fn main() {
    let opts = Opts::parse();
    let mut pixels = Vec::with_capacity(opts.resolution * opts.resolution);

    let start = Instant::now();
    generate(&opts, &mut pixels);
    println!(
        "generated {} samples in {:?} ({:?} per sample)",
        pixels.len(),
        start.elapsed(),
        start.elapsed() / pixels.len() as u32
    );

    println!("encoding...");
    lodepng::encode_file(
        "a.png",
        &pixels,
        opts.resolution,
        opts.resolution,
        lodepng::ColorType::GREY,
        8,
    )
    .unwrap();
}

fn generate(opts: &Opts, pixels: &mut Vec<u8>) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            println!("using AVX2 + FMA");
            unsafe {
                generate_avx2(opts, pixels);
            }
            return;
        } else if is_x86_feature_detected!("sse4.2") {
            println!("using SSE4.2");
            unsafe {
                generate_sse(opts, pixels);
            }
            return;
        }
    }
    println!("no runtime features detected");
    generate_inner::<4>(opts, pixels);
}

#[target_feature(enable = "avx2,fma")]
unsafe fn generate_avx2(opts: &Opts, pixels: &mut Vec<u8>) {
    generate_inner::<8>(opts, pixels);
}

#[target_feature(enable = "sse4.2")]
unsafe fn generate_sse(opts: &Opts, pixels: &mut Vec<u8>) {
    generate_inner::<4>(opts, pixels);
}

#[inline(always)]
fn generate_inner<const LANES: usize>(opts: &Opts, pixels: &mut Vec<u8>)
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let step = opts.scale / opts.resolution as f32;
    for y in 0..opts.resolution {
        let mut x = 0;
        while x < opts.resolution {
            let mut px = Simd::splat(x as f32 * step);
            for i in 1..LANES {
                px[i] += step * i as f32;
            }
            let py = Simd::splat(y as f32 * step);

            let sample = fbm::<LANES>(
                opts.octaves,
                (-opts.hurst_exponent).exp2(),
                opts.lacunarity,
                [px, py],
            );
            let value = (Simd::splat(255.0) * (sample.value + Simd::splat(1.0)) / Simd::splat(2.0))
                .cast::<u8>();
            for i in 0..LANES {
                if x + i >= opts.resolution {
                    break;
                }
                pixels.push(value[i]);
            }
            x += LANES;
        }
    }
}

#[inline(always)]
fn fbm<const LANES: usize>(
    octaves: usize,
    gain: f32,
    lacunarity: f32,
    point: [Simd<f32, LANES>; 2],
) -> Sample<LANES, 2>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    const NOISE: Simplex2d = Simplex2d::new();

    let mut result = Sample::default();

    let mut frequency = 1.0;
    let mut amplitude = 1.0;
    let mut scale = 0.0;
    for _ in 0..octaves {
        result += NOISE.sample(point.map(|x| x * Simd::splat(frequency))) * Simd::splat(amplitude);
        scale += amplitude;
        frequency *= lacunarity;
        amplitude *= gain;
    }
    result / Simd::splat(scale)
}
