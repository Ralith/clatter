use fearless_simd::{Bytes, Select, Simd, SimdBase, SimdFloat, SimdInt};
#[cfg(feature = "rand")]
use rand::{
    distributions::{Distribution, Standard},
    seq::SliceRandom,
    Rng,
};

use crate::{grid, hash, Grid, Sample};

#[derive(Debug, Clone)]
pub struct Simplex1d {
    seed: i32,
}

impl Simplex1d {
    #[inline]
    pub const fn new() -> Self {
        Self { seed: 0 }
    }

    #[cfg(feature = "rand")]
    #[inline]
    pub fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self { seed: rng.gen() }
    }

    #[inline(always)]
    pub fn sample<S: Simd>(&self, point: [S::f32s; 1]) -> Sample<S, 1> {
        let ([[i0], [i1]], [[x0], [x1]]) = grid::Simplex.get::<S>(point);

        // Select gradients
        let gi0 = hash::pcg::<S>((i0 ^ self.seed).bitcast());
        let gi1 = hash::pcg::<S>((i1 ^ self.seed).bitcast());

        // Compute the contribution from the first gradient
        // n0 = grad0 * (1 - x0^2)^4 * x0
        let x20 = x0 * x0;
        let t0 = S::f32s::splat(x0.witness(), 1.0) - x20;
        let t20 = t0 * t0;
        let t40 = t20 * t20;
        let gx0 = gradient_1d::<S>(gi0);
        let n0 = t40 * gx0 * x0;

        // Compute the contribution from the second gradient
        // n1 = grad1 * (x0 - 1) * (1 - (x0 - 1)^2)^4
        let x21 = x1 * x1;
        let t1 = S::f32s::splat(x0.witness(), 1.0) - x21;
        let t21 = t1 * t1;
        let t41 = t21 * t21;
        let gx1 = gradient_1d::<S>(gi1);
        let n1 = t41 * gx1 * x1;

        // n0 + n1 =
        //    grad0 * x0 * (1 - x0^2)^4
        //  + grad1 * (x0 - 1) * (1 - (x0 - 1)^2)^4
        //
        // Assuming worst-case values for grad0 and grad1, we therefore need only determine the maximum of
        //
        // |x0 * (1 - x0^2)^4| + |(x0 - 1) * (1 - (x0 - 1)^2)^4|
        //
        // for 0 ≤ x0 < 1. This can be done by root-finding on the derivative, obtaining 81 / 256 when
        // x0 = 0.5, which we finally multiply by the maximum gradient to get the maximum value,
        // allowing us to scale into [-1, 1]
        const SCALE: f32 = 256.0 / (81.0 * 8.0);
        Sample {
            value: (n0 + n1) * SCALE,
            derivative: [((t20 * t0 * gx0 * x20 + t21 * t1 * gx1 * x21) * -8.0
                + t40 * gx0
                + t41 * gx1)
                * SCALE],
        }
    }
}

impl Default for Simplex1d {
    #[inline]
    fn default() -> Self {
        Simplex1d::new()
    }
}

#[cfg(feature = "rand")]
impl Distribution<Simplex1d> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Simplex1d {
        Simplex1d::random(rng)
    }
}

/// Generates a nonzero random integer gradient in ±8 inclusive
#[inline(always)]
fn gradient_1d<S: Simd>(hash: S::u32s) -> S::f32s {
    let h = hash & 0xF;
    let v = ((h & 7) + 1).to_float::<S::f32s>();

    let h_and_8 = (h & 8).simd_eq(S::u32s::splat(hash.witness(), 0));
    h_and_8.select(v, -v)
}

#[derive(Debug, Clone)]
pub struct Simplex2d {
    seed: i32,
}

impl Simplex2d {
    #[inline]
    pub const fn new() -> Self {
        Self { seed: 0 }
    }

    #[cfg(feature = "rand")]
    #[inline]
    pub fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self { seed: rng.gen() }
    }

    #[inline(always)]
    pub fn sample<S: Simd>(&self, point: [S::f32s; 2]) -> Sample<S, 2> {
        let ([[i0, j0], [i1, j1], [i2, j2]], [[x0, y0], [x1, y1], [x2, y2]]) =
            grid::Simplex.get::<S>(point);

        let seed = S::i32s::splat(i0.witness(), self.seed);
        let gi0 = hash::pcg_3d::<S>([i0, j0, seed].map(|x| x.bitcast()))[0];
        let gi1 = hash::pcg_3d::<S>([i1, j1, seed].map(|x| x.bitcast()))[0];
        let gi2 = hash::pcg_3d::<S>([i2, j2, seed].map(|x| x.bitcast()))[0];

        // Weights associated with the gradients at each corner
        // These FMA operations are equivalent to: let t = max(0, 0.5 - x*x - y*y)
        let t0 = y0.madd(-y0, x0.madd(-x0, 0.5f32)).max(0.0f32);
        let t1 = y1.madd(-y1, x1.madd(-x1, 0.5)).max(0.0);
        let t2 = y2.madd(-y2, x2.madd(-x2, 0.5)).max(0.0);

        let t20 = t0 * t0;
        let t40 = t20 * t20;
        let t21 = t1 * t1;
        let t41 = t21 * t21;
        let t22 = t2 * t2;
        let t42 = t22 * t22;

        let [gx0, gy0] = gradient_2d::<S>(gi0);
        let g0 = gx0 * x0 + gy0 * y0;
        let n0 = t40 * g0;
        let [gx1, gy1] = gradient_2d::<S>(gi1);
        let g1 = gx1 * x1 + gy1 * y1;
        let n1 = t41 * g1;
        let [gx2, gy2] = gradient_2d::<S>(gi2);
        let g2 = gx2 * x2 + gy2 * y2;
        let n2 = t42 * g2;

        // Scaling factor found by numerical optimization
        const SCALE: f32 = 45.26450774985561631259;

        Sample {
            value: (n0 + n1 + n2) * SCALE,
            derivative: {
                let temp0 = t20 * t0 * g0;
                let mut dnoise_dx = temp0 * x0;
                let mut dnoise_dy = temp0 * y0;
                let temp1 = t21 * t1 * g1;
                dnoise_dx += temp1 * x1;
                dnoise_dy += temp1 * y1;
                let temp2 = t22 * t2 * g2;
                dnoise_dx += temp2 * x2;
                dnoise_dy += temp2 * y2;
                dnoise_dx *= -8.0;
                dnoise_dy *= -8.0;
                dnoise_dx += t40 * gx0 + t41 * gx1 + t42 * gx2;
                dnoise_dy += t40 * gy0 + t41 * gy1 + t42 * gy2;
                dnoise_dx *= SCALE;
                dnoise_dy *= SCALE;
                [dnoise_dx, dnoise_dy]
            },
        }
    }
}

impl Default for Simplex2d {
    #[inline]
    fn default() -> Self {
        Simplex2d::new()
    }
}

#[cfg(feature = "rand")]
impl Distribution<Simplex2d> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Simplex2d {
        Simplex2d::random(rng)
    }
}

#[inline(always)]
fn gradient_2d<S: Simd>(hash: S::u32s) -> [S::f32s; 2] {
    let h = hash & 7;

    let mask = h.simd_lt(S::u32s::splat(hash.witness(), 4));
    let _f1 = S::f32s::splat(hash.witness(), 1.0);
    let _f2 = S::f32s::splat(hash.witness(), 2.0);
    let x_magnitude = mask.select(_f1, _f2);
    let y_magnitude = mask.select(_f2, _f1);

    let h_and_1 = (h & 1).simd_eq(0);
    let h_and_2 = (h & 2).simd_eq(0);

    let gx = mask
        .select(h_and_1, h_and_2)
        .select(x_magnitude, -x_magnitude);
    let gy = mask
        .select(h_and_2, h_and_1)
        .select(y_magnitude, -y_magnitude);
    [gx, gy]
}

pub struct Simplex3d {
    seed: i32,
}

impl Simplex3d {
    #[inline]
    pub const fn new() -> Self {
        Self { seed: 0 }
    }

    #[cfg(feature = "rand")]
    #[inline]
    pub fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self { seed: rng.gen() }
    }

    #[inline(always)]
    pub fn sample<S: Simd>(&self, point: [S::f32s; 3]) -> Sample<S, 3> {
        let ([p0, p1, p2, p3], [[x0, y0, z0], [x1, y1, z1], [x2, y2, z2], [x3, y3, z3]]) =
            grid::Simplex.get::<S>(point);

        // Compute base weight factors associated with each vertex, `0.5 - v . v` where v is the
        // difference between the sample point and the vertex.
        let t0 = (-x0 * x0 - y0 * y0 - z0 * z0 + 0.5).max(0.0);
        let t1 = (-x1 * x1 - y1 * y1 - z1 * z1 + 0.5).max(0.0);
        let t2 = (-x2 * x2 - y2 * y2 - z2 * z2 + 0.5).max(0.0);
        let t3 = (-x3 * x3 - y3 * y3 - z3 * z3 + 0.5).max(0.0);

        // Square weights
        let t20 = t0 * t0;
        let t21 = t1 * t1;
        let t22 = t2 * t2;
        let t23 = t3 * t3;

        // ...twice!
        let t40 = t20 * t20;
        let t41 = t21 * t21;
        let t42 = t22 * t22;
        let t43 = t23 * t23;

        // Compute contribution from each vertex
        let g0 = Gradient3d::<S>::new(self.seed, p0);
        let g0d = g0.dot([x0, y0, z0]);
        let v0 = t40 * g0d;

        let g1 = Gradient3d::<S>::new(self.seed, p1);
        let g1d = g1.dot([x1, y1, z1]);
        let v1 = t41 * g1d;

        let g2 = Gradient3d::<S>::new(self.seed, p2);
        let g2d = g2.dot([x2, y2, z2]);
        let v2 = t42 * g2d;

        let g3 = Gradient3d::<S>::new(self.seed, p3);
        let g3d = g3.dot([x3, y3, z3]);
        let v3 = t43 * g3d;

        // Scaling factor found by numerical optimization
        const SCALE: f32 = 67.79816627147162;
        Sample {
            value: (v3 + v2 + v1 + v0) * SCALE,
            derivative: {
                let temp0 = t20 * t0 * g0d;
                let mut dnoise_dx = temp0 * x0;
                let mut dnoise_dy = temp0 * y0;
                let mut dnoise_dz = temp0 * z0;
                let temp1 = t21 * t1 * g1d;
                dnoise_dx += temp1 * x1;
                dnoise_dy += temp1 * y1;
                dnoise_dz += temp1 * z1;
                let temp2 = t22 * t2 * g2d;
                dnoise_dx += temp2 * x2;
                dnoise_dy += temp2 * y2;
                dnoise_dz += temp2 * z2;
                let temp3 = t23 * t3 * g3d;
                dnoise_dx += temp3 * x3;
                dnoise_dy += temp3 * y3;
                dnoise_dz += temp3 * z3;
                dnoise_dx *= -8.0;
                dnoise_dy *= -8.0;
                dnoise_dz *= -8.0;
                let [gx0, gy0, gz0] = g0.vector();
                let [gx1, gy1, gz1] = g1.vector();
                let [gx2, gy2, gz2] = g2.vector();
                let [gx3, gy3, gz3] = g3.vector();
                dnoise_dx += t40 * gx0 + t41 * gx1 + t42 * gx2 + t43 * gx3;
                dnoise_dy += t40 * gy0 + t41 * gy1 + t42 * gy2 + t43 * gy3;
                dnoise_dz += t40 * gz0 + t41 * gz1 + t42 * gz2 + t43 * gz3;
                // Scale into range
                dnoise_dx *= SCALE;
                dnoise_dy *= SCALE;
                dnoise_dz *= SCALE;
                [dnoise_dx, dnoise_dy, dnoise_dz]
            },
        }
    }
}

impl Default for Simplex3d {
    #[inline]
    fn default() -> Self {
        Simplex3d::new()
    }
}

#[cfg(feature = "rand")]
impl Distribution<Simplex3d> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Simplex3d {
        Simplex3d::random(rng)
    }
}

/// Generates a random gradient vector from the origin towards the midpoint of an edge of a
/// double-unit cube
struct Gradient3d<S: Simd> {
    // Masks guiding dimension selection
    l8: S::mask32s,
    l4: S::mask32s,
    h12_or_14: S::mask32s,

    // Signs for the selected dimensions
    h1: S::u32s,
    h2: S::u32s,
}

impl<S: Simd> Gradient3d<S> {
    /// Compute hash values used by `grad3d` and `grad3d_dot`
    #[inline(always)]
    fn new(seed: i32, [i, j, k]: [S::i32s; 3]) -> Self {
        let hash =
            hash::pcg_4d::<S>([i, j, k, S::i32s::splat(i.witness(), seed)].map(|x| x.bitcast()))[0];
        let hasha13 = hash & 13;
        Self {
            l8: hasha13.simd_lt(8),
            l4: hasha13.simd_lt(2),
            h12_or_14: hasha13.simd_eq(12),

            h1: hash << 31,
            h2: (hash & 2) << 30,
        }
    }

    /// Computes the dot product of a vector with the gradient vector
    #[inline(always)]
    fn dot(&self, [x, y, z]: [S::f32s; 3]) -> S::f32s {
        let u = self.l8.select(x, y);
        let v = self.l4.select(y, self.h12_or_14.select(x, z));
        // Maybe flip sign bits, then sum
        (u.bitcast::<S::u32s>() ^ self.h1).bitcast::<S::f32s>()
            + (v.bitcast::<S::u32s>() ^ self.h2).bitcast::<S::f32s>()
    }

    /// The gradient vector generated by `dot`
    ///
    /// This is a separate function because it's slower than `grad3d_dot` and only needed when computing
    /// derivatives.
    #[inline(always)]
    fn vector(&self) -> [S::f32s; 3] {
        let first = (self.h1 | S::f32s::splat(self.h1.witness(), 1.0).bitcast::<S::u32s>())
            .bitcast::<S::f32s>();
        let gx = self
            .l8
            .select(first, S::f32s::splat(self.h1.witness(), 0.0));
        let gy = (!self.l8).select(first, S::f32s::splat(self.h1.witness(), 0.0));

        let second = (self.h2 | S::f32s::splat(self.h1.witness(), 1.0).bitcast::<S::u32s>())
            .bitcast::<S::f32s>();
        let gy = self.l4.select(second, gy);
        let gx = (!self.l4 & self.h12_or_14).select(second, gx);
        let gz =
            (!(self.h12_or_14 | self.l4)).select(second, S::f32s::splat(self.h1.witness(), 0.0));

        [gx, gy, gz]
    }
}

pub struct Simplex4d {
    seed: i32,
}

impl Simplex4d {
    #[inline]
    pub const fn new() -> Self {
        Self { seed: 0 }
    }

    #[cfg(feature = "rand")]
    #[inline]
    pub fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self { seed: rng.gen() }
    }

    #[inline(always)]
    pub fn sample<S: Simd>(&self, point: [S::f32s; 4]) -> Sample<S, 4> {
        let (
            [p0, p1, p2, p3, p4],
            [[x0, y0, z0, w0], [x1, y1, z1, w1], [x2, y2, z2, w2], [x3, y3, z3, w3], [x4, y4, z4, w4]],
        ) = grid::Simplex.get::<S>(point);

        //
        // Hash the integer coordinates
        //

        let gi0 = Gradient4d::<S>::new(self.seed, p0);
        let gi1 = Gradient4d::<S>::new(self.seed, p1);
        let gi2 = Gradient4d::<S>::new(self.seed, p2);
        let gi3 = Gradient4d::<S>::new(self.seed, p3);
        let gi4 = Gradient4d::<S>::new(self.seed, p4);

        //
        // Compute base weight factors associated with each vertex
        //

        // These FMA operations are equivalent to: let t = max(0, 0.5 - x*x - y*y - z*z - w*w)
        let t0 = w0
            .madd(-w0, z0.madd(-z0, y0.madd(-y0, x0.madd(-x0, 0.5))))
            .max(0.0);
        let t1 = w1
            .madd(-w1, z1.madd(-z1, y1.madd(-y1, x1.madd(-x1, 0.5))))
            .max(0.0);
        let t2 = w2
            .madd(-w2, z2.madd(-z2, y2.madd(-y2, x2.madd(-x2, 0.5))))
            .max(0.0);
        let t3 = w3
            .madd(-w3, z3.madd(-z3, y3.madd(-y3, x3.madd(-x3, 0.5))))
            .max(0.0);
        let t4 = w4
            .madd(-w4, z4.madd(-z4, y4.madd(-y4, x4.madd(-x4, 0.5))))
            .max(0.0);

        // Cube each weight
        let t02 = t0 * t0;
        let t04 = t02 * t02;
        let t12 = t1 * t1;
        let t14 = t12 * t12;
        let t22 = t2 * t2;
        let t24 = t22 * t22;
        let t32 = t3 * t3;
        let t34 = t32 * t32;
        let t42 = t4 * t4;
        let t44 = t42 * t42;

        // Compute contributions from each gradient
        let g0d = gi0.dot([x0, y0, z0, w0]);
        let g1d = gi1.dot([x1, y1, z1, w1]);
        let g2d = gi2.dot([x2, y2, z2, w2]);
        let g3d = gi3.dot([x3, y3, z3, w3]);
        let g4d = gi4.dot([x4, y4, z4, w4]);

        let n0 = t04 * g0d;
        let n1 = t14 * g1d;
        let n2 = t24 * g2d;
        let n3 = t34 * g3d;
        let n4 = t44 * g4d;

        // Scaling factor found by numerical optimization
        const SCALE: f32 = 62.77772078955791;
        Sample {
            value: (n0 + n1 + n2 + n3 + n4) * SCALE,
            derivative: {
                let temp = t02 * t0 * g0d;
                let mut dvdx = temp * x0;
                let mut dvdy = temp * y0;
                let mut dvdz = temp * z0;
                let mut dvdw = temp * w0;

                let temp = t12 * t1 * g1d;
                dvdx += temp * x1;
                dvdy += temp * y1;
                dvdz += temp * z1;
                dvdw += temp * w1;

                let temp = t22 * t2 * g2d;
                dvdx += temp * x2;
                dvdy += temp * y2;
                dvdz += temp * z2;
                dvdw += temp * w2;

                let temp = t32 * t3 * g3d;
                dvdx += temp * x3;
                dvdy += temp * y3;
                dvdz += temp * z3;
                dvdw += temp * w3;

                let temp = t42 * t4 * g4d;
                dvdx += temp * x4;
                dvdy += temp * y4;
                dvdz += temp * z4;
                dvdw += temp * w4;

                dvdx *= -8.0;
                dvdy *= -8.0;
                dvdz *= -8.0;
                dvdw *= -8.0;

                let g0 = gi0.vector();
                let g1 = gi1.vector();
                let g2 = gi2.vector();
                let g3 = gi3.vector();
                let g4 = gi4.vector();
                dvdx += t04 * g0[0] + t14 * g1[0] + t24 * g2[0] + t34 * g3[0] + t44 * g4[0];
                dvdy += t04 * g0[1] + t14 * g1[1] + t24 * g2[1] + t34 * g3[1] + t44 * g4[1];
                dvdz += t04 * g0[2] + t14 * g1[2] + t24 * g2[2] + t34 * g3[2] + t44 * g4[2];
                dvdw += t04 * g0[3] + t14 * g1[3] + t24 * g2[3] + t34 * g3[3] + t44 * g4[3];

                [dvdx, dvdy, dvdz, dvdw].map(|x| x * SCALE)
            },
        }
    }
}

impl Default for Simplex4d {
    #[inline]
    fn default() -> Self {
        Simplex4d::new()
    }
}

#[cfg(feature = "rand")]
impl Distribution<Simplex4d> for Standard {
    #[inline]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Simplex4d {
        Simplex4d::random(rng)
    }
}

/// Uniformly maps i32s to vectors from the origin towards the midpoint of an edge of a hypercube
struct Gradient4d<S: Simd> {
    // Masks guiding dimension selection
    l24: S::mask32s,
    l16: S::mask32s,
    l8: S::mask32s,

    // Signs for the selected dimensions
    sign1: S::mask32s,
    sign2: S::mask32s,
    sign3: S::mask32s,
}

impl<S: Simd> Gradient4d<S> {
    #[inline(always)]
    fn new(seed: i32, [i, j, k, l]: [S::i32s; 4]) -> Self {
        let hash = hash::pcg_4d::<S>(
            [i ^ seed, j ^ seed, k ^ seed, l ^ seed].map(|x| x.bitcast::<S::u32s>()),
        )[0];

        let h = hash & 31;
        Self {
            l24: S::u32s::splat(i.witness(), 24).simd_gt(h),
            l16: S::u32s::splat(i.witness(), 16).simd_gt(h),
            l8: S::u32s::splat(i.witness(), 8).simd_gt(h),

            sign1: S::u32s::splat(i.witness(), 0).simd_eq(h & 1),
            sign2: S::u32s::splat(i.witness(), 0).simd_eq(h & 2),
            sign3: S::u32s::splat(i.witness(), 0).simd_eq(h & 4),
        }
    }

    /// Directly compute the dot product of the gradient vector with a vector
    #[inline(always)]
    fn dot(&self, [x, y, z, t]: [S::f32s; 4]) -> S::f32s {
        let u = self.l24.select(x, y);
        let v = self.l16.select(y, z);
        let w = self.l8.select(z, t);
        self.sign1.select(u, -u) + self.sign2.select(v, -v) + self.sign3.select(w, -w)
    }

    /// Compute the actual gradient vector
    ///
    /// Slower than `dot` and only needed to compute derivatives
    #[inline(always)]
    fn vector(&self) -> [S::f32s; 4] {
        // Select axes
        //       h: u  v  w
        // 24..=31: y, z, t
        // 17..=23: x, z, t
        //  8..=16: x, y, t
        //  0..=7 : x, y, z
        let gx = self
            .l24
            .select(
                S::u32s::splat(self.l8.witness(), 1),
                S::u32s::splat(self.l8.witness(), 0),
            )
            .bitcast::<S::i32s>();
        let gy = (self.l16 | !self.l24)
            .select(
                S::u32s::splat(self.l8.witness(), 1),
                S::u32s::splat(self.l8.witness(), 0),
            )
            .bitcast::<S::i32s>();
        let gz = (self.l8 | !self.l16)
            .select(
                S::u32s::splat(self.l8.witness(), 1),
                S::u32s::splat(self.l8.witness(), 0),
            )
            .bitcast::<S::i32s>();
        let gt = (!self.l8)
            .select(
                S::u32s::splat(self.l8.witness(), 1),
                S::u32s::splat(self.l8.witness(), 0),
            )
            .bitcast::<S::i32s>();

        // Select signs
        let gx = self.sign1.select(gx, gx * -1);
        let gy = self.l24.select(self.sign2, self.sign1).select(gy, gy * -1);
        let gz = self.l16.select(self.sign3, self.sign2).select(gz, gz * -1);
        let gt = self.sign3.select(gt, gt * -1);

        [gx, gy, gz, gt].map(|x| x.to_float())
    }
}

#[cfg(test)]
mod tests {
    use fearless_simd::{f32x4, i32x4, Fallback};

    use super::*;

    type S = Fallback;
    const SIMD: S = Fallback::new();

    fn check_bounds(min: f32, max: f32) {
        dbg!(min, max);
        assert!(min < -0.75 && min >= -1.0, "min out of range {}", min);
        assert!(max > 0.75 && max <= 1.0, "max out of range: {}", max);
    }

    #[test]
    fn simplex_1d_range() {
        const G: Simplex1d = Simplex1d::new();

        let mut min = f32::INFINITY;
        let mut max = -f32::INFINITY;
        for x in 0..1000 {
            let n = G.sample::<S>([f32x4::splat(SIMD, x as f32 / 10.0)]).value[0];
            min = min.min(n);
            max = max.max(n);
        }
        check_bounds(min, max);
    }

    #[test]
    fn simplex_1d_deriv_sanity() {
        const G: Simplex1d = Simplex1d::new();
        let mut avg_err = 0.0;
        const SEEDS: i32 = 10;
        const POINTS: i32 = 1000;
        for x in 0..POINTS {
            // Offset a bit so we don't check derivative at lattice points, where it's always zero
            let center = x as f32 / 10.0 + 0.1234;
            const H: f32 = 0.01;
            let n0 = G.sample::<S>([f32x4::splat(SIMD, center - H)]).value[0];
            let Sample {
                value: n1,
                derivative: d1,
            } = G.sample::<S>([f32x4::splat(SIMD, center)]);
            let n2 = G.sample::<S>([f32x4::splat(SIMD, center + H)]).value[0];
            let (n1, d1) = (n1[0], d1[0][0]);
            avg_err += ((n2 - (n1 + d1 * H)).abs() + (n0 - (n1 - d1 * H)).abs())
                / (SEEDS * POINTS * 2) as f32;
        }
        assert!(avg_err < 1e-3);
    }

    #[test]
    fn simplex_2d_range() {
        const G: Simplex2d = Simplex2d::new();
        let mut min = f32::INFINITY;
        let mut max = -f32::INFINITY;
        for y in 0..10 {
            for x in 0..100 {
                let n = G
                    .sample::<S>([
                        f32x4::splat(SIMD, x as f32 / 10.0),
                        f32x4::splat(SIMD, y as f32 / 10.0),
                    ])
                    .value[0];
                println!("{},{} = {}", x, y, n);
                min = min.min(n);
                max = max.max(n);
            }
        }
        check_bounds(min, max);
    }

    #[test]
    fn simplex_2d_deriv_sanity() {
        const G: Simplex2d = Simplex2d::new();
        let mut avg_err = 0.0;
        const POINTS: i32 = 10;
        for y in 0..POINTS {
            for x in 0..POINTS {
                // Offset a bit so we don't check derivative at lattice points, where it's always zero
                let center_x = x as f32 / 10.0 + 0.1234;
                let center_y = y as f32 / 10.0 + 0.1234;
                const H: f32 = 0.01;
                let Sample {
                    value,
                    derivative: d,
                } = G.sample::<S>([f32x4::splat(SIMD, center_x), f32x4::splat(SIMD, center_y)]);
                let (value, d) = (value[0], [d[0][0], d[1][0]]);
                let left = G
                    .sample::<S>([
                        f32x4::splat(SIMD, center_x - H),
                        f32x4::splat(SIMD, center_y),
                    ])
                    .value[0];
                let right = G
                    .sample::<S>([
                        f32x4::splat(SIMD, center_x + H),
                        f32x4::splat(SIMD, center_y),
                    ])
                    .value[0];
                let down = G
                    .sample::<S>([
                        f32x4::splat(SIMD, center_x),
                        f32x4::splat(SIMD, center_y - H),
                    ])
                    .value[0];
                let up = G
                    .sample::<S>([
                        f32x4::splat(SIMD, center_x),
                        f32x4::splat(SIMD, center_y + H),
                    ])
                    .value[0];
                avg_err += ((left - (value - d[0] * H)).abs()
                    + (right - (value + d[0] * H)).abs()
                    + (down - (value - d[1] * H)).abs()
                    + (up - (value + d[1] * H)).abs())
                    / (POINTS * POINTS * 4) as f32;
            }
        }
        assert!(avg_err < 1e-3);
    }

    #[test]
    fn gradient_3d_consistency() {
        for k in 0..10 {
            for j in 0..10 {
                for i in 0..10 {
                    let vertex = [
                        i32x4::splat(SIMD, i),
                        i32x4::splat(SIMD, j),
                        i32x4::splat(SIMD, k),
                    ];
                    let g = Gradient3d::<S>::new(0, vertex);
                    let grad = g.vector();
                    assert_eq!(
                        g.dot([
                            f32x4::splat(SIMD, 1.0),
                            f32x4::splat(SIMD, 10.0),
                            f32x4::splat(SIMD, 100.0)
                        ])[0],
                        grad[0][0] + 10.0 * grad[1][0] + 100.0 * grad[2][0]
                    );
                }
            }
        }
    }

    #[test]
    fn simplex_3d_range() {
        const G: Simplex3d = Simplex3d::new();
        let mut min = f32::INFINITY;
        let mut max = -f32::INFINITY;
        for z in 0..10 {
            for y in 0..10 {
                for x in 0..1000 {
                    let n = G
                        .sample::<S>([
                            f32x4::splat(SIMD, x as f32 / 10.0),
                            f32x4::splat(SIMD, y as f32 / 10.0),
                            f32x4::splat(SIMD, z as f32 / 10.0),
                        ])
                        .value[0];
                    min = min.min(n);
                    max = max.max(n);
                }
            }
        }
        check_bounds(min, max);
    }

    #[test]
    fn simplex_3d_deriv_sanity() {
        const G: Simplex3d = Simplex3d::new();
        let mut avg_err = 0.0;
        const POINTS: i32 = 20;
        for z in 0..POINTS {
            for y in 0..POINTS {
                for x in 0..POINTS {
                    // Offset a bit so we don't check derivative at lattice points, where it's always zero
                    let center_x = x as f32 / 3.0 + 0.1234;
                    let center_y = y as f32 / 3.0 + 0.1234;
                    let center_z = z as f32 / 3.0 + 0.1234;
                    const H: f32 = 0.01;
                    let Sample {
                        value,
                        derivative: d,
                    } = G.sample::<S>([
                        f32x4::splat(SIMD, center_x),
                        f32x4::splat(SIMD, center_y),
                        f32x4::splat(SIMD, center_z),
                    ]);
                    let (value, d) = (value[0], [d[0][0], d[1][0], d[2][0]]);
                    let right = G
                        .sample::<S>([
                            f32x4::splat(SIMD, center_x + H),
                            f32x4::splat(SIMD, center_y),
                            f32x4::splat(SIMD, center_z),
                        ])
                        .value[0];
                    let up = G
                        .sample::<S>([
                            f32x4::splat(SIMD, center_x),
                            f32x4::splat(SIMD, center_y + H),
                            f32x4::splat(SIMD, center_z),
                        ])
                        .value[0];
                    let forward = G
                        .sample::<S>([
                            f32x4::splat(SIMD, center_x),
                            f32x4::splat(SIMD, center_y),
                            f32x4::splat(SIMD, center_z + H),
                        ])
                        .value[0];
                    dbg!(value, d, right, up, forward);
                    avg_err += ((right - (value + d[0] * H)).abs()
                        + (up - (value + d[1] * H)).abs()
                        + (forward - (value + d[2] * H)).abs())
                        / (POINTS * POINTS * POINTS * 3) as f32;
                }
            }
        }
        assert!(avg_err < 1e-3);
    }

    #[test]
    fn gradient_4d_consistency() {
        for l in 0..10 {
            for k in 0..10 {
                for j in 0..10 {
                    for i in 0..10 {
                        let vertex = [
                            i32x4::splat(SIMD, i),
                            i32x4::splat(SIMD, j),
                            i32x4::splat(SIMD, k),
                            i32x4::splat(SIMD, l),
                        ];
                        let g = Gradient4d::<S>::new(0, vertex);
                        let grad = g.vector();
                        assert_eq!(
                            g.dot([
                                f32x4::splat(SIMD, 1.0),
                                f32x4::splat(SIMD, 10.0),
                                f32x4::splat(SIMD, 100.0),
                                f32x4::splat(SIMD, 1000.0)
                            ])[0],
                            grad[0][0]
                                + 10.0 * grad[1][0]
                                + 100.0 * grad[2][0]
                                + 1000.0 * grad[3][0]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn simplex_4d_range() {
        const G: Simplex4d = Simplex4d::new();
        let mut min = f32::INFINITY;
        let mut max = -f32::INFINITY;
        for w in 0..10 {
            for z in 0..10 {
                for y in 0..10 {
                    for x in 0..100 {
                        let n = G
                            .sample::<S>([
                                f32x4::splat(SIMD, x as f32 / 10.0),
                                f32x4::splat(SIMD, y as f32 / 10.0),
                                f32x4::splat(SIMD, z as f32 / 10.0),
                                f32x4::splat(SIMD, w as f32 / 10.0),
                            ])
                            .value[0];
                        min = min.min(n);
                        max = max.max(n);
                    }
                }
            }
        }
        check_bounds(min, max);
    }

    #[test]
    fn simplex_4d_deriv_sanity() {
        const G: Simplex4d = Simplex4d::new();
        const POINTS: i32 = 10;
        for w in 0..POINTS {
            for z in 0..POINTS {
                for y in 0..POINTS {
                    for x in 0..POINTS {
                        // Offset a bit so we don't check derivative at lattice points, where it's always zero
                        let center_x = x as f32 / 3.0 + 0.1234;
                        let center_y = y as f32 / 3.0 + 0.1234;
                        let center_z = z as f32 / 3.0 + 0.1234;
                        let center_w = w as f32 / 3.0 + 0.1234;
                        println!("({}, {}, {}, {})", center_x, center_y, center_z, center_w);
                        const H: f32 = 0.005;
                        let Sample {
                            value,
                            derivative: d,
                        } = G.sample::<S>([
                            f32x4::splat(SIMD, center_x),
                            f32x4::splat(SIMD, center_y),
                            f32x4::splat(SIMD, center_z),
                            f32x4::splat(SIMD, center_w),
                        ]);
                        let (value, d) = (value[0], [d[0][0], d[1][0], d[2][0], d[3][0]]);
                        let right = G
                            .sample::<S>([
                                f32x4::splat(SIMD, center_x + H),
                                f32x4::splat(SIMD, center_y),
                                f32x4::splat(SIMD, center_z),
                                f32x4::splat(SIMD, center_w),
                            ])
                            .value[0];
                        let up = G
                            .sample::<S>([
                                f32x4::splat(SIMD, center_x),
                                f32x4::splat(SIMD, center_y + H),
                                f32x4::splat(SIMD, center_z),
                                f32x4::splat(SIMD, center_w),
                            ])
                            .value[0];
                        let forward = G
                            .sample::<S>([
                                f32x4::splat(SIMD, center_x),
                                f32x4::splat(SIMD, center_y),
                                f32x4::splat(SIMD, center_z + H),
                                f32x4::splat(SIMD, center_w),
                            ])
                            .value[0];
                        let fourthward = G
                            .sample::<S>([
                                f32x4::splat(SIMD, center_x),
                                f32x4::splat(SIMD, center_y),
                                f32x4::splat(SIMD, center_z),
                                f32x4::splat(SIMD, center_w + H),
                            ])
                            .value[0];
                        let approx = [right, up, forward, fourthward].map(|x| (x - value) / H);
                        println!("analytic = {:+.5?}\nnumeric  = {:+.5?}\n", d, approx);
                        for (a, b) in d.into_iter().zip(approx) {
                            assert!((b - a).abs() < 0.1);
                        }
                    }
                }
            }
        }
    }
}
