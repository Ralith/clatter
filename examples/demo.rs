#![feature(portable_simd)]

use std::{simd::Simd, time::Instant};

use clap::Parser;

use clatter::Simplex4d;

#[derive(Parser)]
struct Opts {
    #[clap(short, default_value = "5.0")]
    scale: f32,
    #[clap(short, default_value = "512")]
    resolution: usize,
}

const N: usize = 8;

fn main() {
    let opts = Opts::parse();
    let step = opts.scale / opts.resolution as f32;
    let noise = Simplex4d::new();
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
            let pz = Simd::splat(0.5);
            let pw = Simd::splat(0.7);

            let sample = noise.sample::<N>([px, py, pz, pw]);
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
