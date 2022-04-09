#![feature(portable_simd)]

use std::{
    simd::{LaneCount, Simd, SupportedLaneCount},
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
    #[clap(short, default_value = "1.0")]
    hurst_exponent: f32,
    /// Frequency ratio between successive octaves
    #[clap(short, default_value = "1.9")]
    lacunarity: f32,
}

const N: usize = 8;

fn main() {
    let opts = Opts::parse();
    let step = opts.scale / opts.resolution as f32;
    let mut pixels = Vec::with_capacity(opts.resolution * opts.resolution);

    let start = Instant::now();
    for y in 0..opts.resolution {
        let mut x = 0;
        while x < opts.resolution {
            let mut px = Simd::splat(x as f32 * step);
            for i in 1..N {
                px[i] += step * i as f32;
            }
            let py = Simd::splat(y as f32 * step);

            let sample = fbm::<N>(
                opts.octaves,
                (-opts.hurst_exponent).exp2(),
                opts.lacunarity,
                [px, py],
            );
            let value = (Simd::splat(255.0) * (sample.value + Simd::splat(1.0)) / Simd::splat(2.0))
                .cast::<u8>();
            for i in 0..N {
                if x + i >= opts.resolution {
                    break;
                }
                pixels.push(value[i]);
            }
            x += N;
        }
    }
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
