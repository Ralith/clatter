use std::array;

use bencher::{benchmark_group, benchmark_main, black_box, Bencher};
use clatter::{Simplex1d, Simplex2d, Simplex3d};
use fearless_simd::{Level, Simd, SimdBase, WithSimd};

fn simplex_1d(bench: &mut Bencher) {
    let level = Level::new();
    bench.iter(|| {
        let x = array::from_fn::<f32, 8, _>(|i| black_box(0.5 + i as f32 * 0.2));
        let mut out = [0.0; 8];
        level.dispatch(Sample::<1> {
            x: [&x],
            out: &mut out,
        });
        black_box(out);
    });
}

fn simplex_2d(bench: &mut Bencher) {
    let level = Level::new();
    bench.iter(|| {
        let x = array::from_fn::<f32, 8, _>(|i| black_box(0.5 + i as f32 * 0.2));
        let y = x;
        let mut out = [0.0; 8];
        level.dispatch(Sample::<2> {
            x: [&x, &y],
            out: &mut out,
        });
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
        level.dispatch(Sample::<3> {
            x: [&x, &y, &z],
            out: &mut out,
        });
        black_box(out);
    });
}

benchmark_group!(benches, simplex_1d, simplex_2d, simplex_3d);
benchmark_main!(benches);

struct Sample<'a, 'b, const DIM: usize> {
    x: [&'a [f32]; DIM],
    out: &'b mut [f32],
}

impl<'a, 'b> WithSimd for Sample<'a, 'b, 1> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        const G: Simplex1d = Simplex1d::new();
        for (chunk, out) in self.x[0]
            .chunks_exact(S::f32s::N)
            .zip(self.out.chunks_exact_mut(S::f32s::N))
        {
            out.copy_from_slice(
                G.sample::<S>([S::f32s::from_slice(simd, chunk)])
                    .value
                    .as_slice(),
            )
        }
    }
}

impl<'a, 'b> WithSimd for Sample<'a, 'b, 2> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        const G: Simplex2d = Simplex2d::new();
        for ((x, y), out) in self.x[0]
            .chunks_exact(S::f32s::N)
            .zip(self.x[1].chunks_exact(S::f32s::N))
            .zip(self.out.chunks_exact_mut(S::f32s::N))
        {
            out.copy_from_slice(
                G.sample::<S>([S::f32s::from_slice(simd, x), S::f32s::from_slice(simd, y)])
                    .value
                    .as_slice(),
            )
        }
    }
}

impl<'a, 'b> WithSimd for Sample<'a, 'b, 3> {
    type Output = ();
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) {
        const G: Simplex3d = Simplex3d::new();
        for (((x, y), z), out) in self.x[0]
            .chunks_exact(S::f32s::N)
            .zip(self.x[1].chunks_exact(S::f32s::N))
            .zip(self.x[2].chunks_exact(S::f32s::N))
            .zip(self.out.chunks_exact_mut(S::f32s::N))
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
}
