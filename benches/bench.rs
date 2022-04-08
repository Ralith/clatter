#![feature(portable_simd)]

use std::simd::Simd;

use bencher::{benchmark_group, benchmark_main, black_box, Bencher};

fn simplex_1d(bench: &mut Bencher) {
    let g = clatter::Simplex1d::new();
    bench.iter(|| black_box(g.sample::<8>([Simd::splat(42.5)])))
}

fn simplex_2d(bench: &mut Bencher) {
    let g = clatter::Simplex2d::new();
    bench.iter(|| black_box(g.sample::<8>([Simd::splat(42.5), Simd::splat(17.5)])))
}

fn simplex_3d(bench: &mut Bencher) {
    let g = clatter::Simplex3d::new();
    bench.iter(|| {
        black_box(g.sample::<8>([Simd::splat(42.5), Simd::splat(17.5), Simd::splat(12.5)]))
    })
}

fn simplex_4d(bench: &mut Bencher) {
    let g = clatter::Simplex4d::new();
    bench.iter(|| {
        black_box(g.sample::<8>([
            Simd::splat(42.5),
            Simd::splat(17.5),
            Simd::splat(12.5),
            Simd::splat(87.1),
        ]))
    })
}

benchmark_group!(benches, simplex_1d, simplex_2d, simplex_3d, simplex_4d);
benchmark_main!(benches);
