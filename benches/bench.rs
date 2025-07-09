use std::array;

use bencher::{benchmark_group, benchmark_main, black_box, Bencher};
use clatter::{Simplex1d, Simplex2d, Simplex3d};
use fearless_simd::{dispatch, Level, Simd, SimdBase};

fn simplex_1d(bench: &mut Bencher) {
    let level = Level::new();
    bench.iter(|| {
        let x = array::from_fn::<f32, 8, _>(|i| black_box(0.5 + i as f32 * 0.2));
        let mut out = [0.0; 8];
        dispatch!(level, simd => sample_1d(simd, &x, &mut out));
        black_box(out);
    });
}

fn simplex_2d(bench: &mut Bencher) {
    let level = Level::new();
    bench.iter(|| {
        let x = array::from_fn::<f32, 8, _>(|i| black_box(0.5 + i as f32 * 0.2));
        let y = x;
        let mut out = [0.0; 8];
        dispatch!(level, simd => sample_2d(simd, [&x, &y], &mut out));
        black_box(out);
    });
}

fn simplex_3d(bench: &mut Bencher) {
    let level = Level::new();
    bench.iter(|| {
        let x = array::from_fn::<f32, 8, _>(|i| black_box(0.5 + i as f32 * 0.2));
        let y = x;
        let z = x;
        let mut out = [0.0; 8];
        dispatch!(level, simd => sample_3d(simd, [&x, &y, &z], &mut out));
        black_box(out);
    });
}

benchmark_group!(benches, simplex_1d, simplex_2d, simplex_3d);
benchmark_main!(benches);

#[inline(always)]
fn sample_1d<S: Simd>(simd: S, x: &[f32], out: &mut [f32]) {
    const G: Simplex1d = Simplex1d::new();
    for (chunk, out) in x
        .chunks_exact(S::f32s::N)
        .zip(out.chunks_exact_mut(S::f32s::N))
    {
        out.copy_from_slice(
            G.sample::<S>([S::f32s::from_slice(simd, chunk)])
                .value
                .as_slice(),
        )
    }
}

#[inline(always)]
fn sample_2d<S: Simd>(simd: S, x: [&[f32]; 2], out: &mut [f32]) {
    const G: Simplex2d = Simplex2d::new();
    for ((x, y), out) in x[0]
        .chunks_exact(S::f32s::N)
        .zip(x[1].chunks_exact(S::f32s::N))
        .zip(out.chunks_exact_mut(S::f32s::N))
    {
        out.copy_from_slice(
            G.sample::<S>([S::f32s::from_slice(simd, x), S::f32s::from_slice(simd, y)])
                .value
                .as_slice(),
        )
    }
}

#[inline(always)]
fn sample_3d<S: Simd>(simd: S, x: [&[f32]; 3], out: &mut [f32]) {
    const G: Simplex3d = Simplex3d::new();
    for (((x, y), z), out) in x[0]
        .chunks_exact(S::f32s::N)
        .zip(x[1].chunks_exact(S::f32s::N))
        .zip(x[2].chunks_exact(S::f32s::N))
        .zip(out.chunks_exact_mut(S::f32s::N))
    {
        out.copy_from_slice(
            G.sample::<S>([
                S::f32s::from_slice(simd, x),
                S::f32s::from_slice(simd, y),
                S::f32s::from_slice(simd, z),
            ])
            .value
            .as_slice(),
        )
    }
}
