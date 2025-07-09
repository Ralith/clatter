use std::time::Instant;

use clap::Parser;

use clatter::{Sample, Simplex2d};
use fearless_simd::{dispatch, Simd, SimdBase, SimdFloat};

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
    let level = fearless_simd::Level::new();
    dispatch!(level, simd => generate(simd, &opts, &mut pixels));
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

fn generate<S: Simd>(simd: S, opts: &Opts, pixels: &mut Vec<u8>) {
    let step = opts.scale / opts.resolution as f32;
    for y in 0..opts.resolution {
        let mut x = 0;
        while x < opts.resolution {
            let mut px = S::f32s::splat(simd, x as f32 * step);
            for i in 1..S::f32s::N {
                px.as_mut_slice()[i] += step * i as f32;
            }
            let py = S::f32s::splat(simd, y as f32 * step);

            let sample = fbm::<S>(
                opts.octaves,
                (-opts.hurst_exponent).exp2(),
                opts.lacunarity,
                [px, py],
            );
            let value = ((sample.value + 1.0) * 255.0 / 2.0).to_int::<S::u32s>();
            for i in 0..S::f32s::N {
                if x + i >= opts.resolution {
                    break;
                }
                pixels.push(value.as_slice()[i] as u8);
            }
            x += S::f32s::N;
        }
    }
}

#[inline(always)]
fn fbm<S: Simd>(octaves: usize, gain: f32, lacunarity: f32, point: [S::f32s; 2]) -> Sample<S, 2> {
    const NOISE: Simplex2d = Simplex2d::new();
    let simd = point[0].witness();

    let mut result = Sample::constant(S::f32s::splat(simd, 0.0));

    let mut frequency = 1.0;
    let mut amplitude = 1.0;
    let mut scale = 0.0;
    for _ in 0..octaves {
        result += NOISE.sample::<S>(point.map(|x| x * frequency)) * S::f32s::splat(simd, amplitude);
        scale += amplitude;
        frequency *= lacunarity;
        amplitude *= gain;
    }
    result / S::f32s::splat(simd, scale)
}
