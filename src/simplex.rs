use std::simd::{
    LaneCount, Mask, Simd, SimdFloat, SimdPartialEq, SimdPartialOrd, StdFloat as _,
    SupportedLaneCount,
};

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
    pub fn sample<const LANES: usize>(&self, point: [Simd<f32, LANES>; 1]) -> Sample<LANES, 1>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let (is, xs) = grid::Simplex.get(point);

        let mut value = Simd::splat(0.0);
        let mut diff_a = Simd::splat(0.0);
        let mut diff_b = Simd::splat(0.0);

        for v in 0..2 {
            let [i] = is[v];
            let [dx] = xs[v];

            // Select gradient
            let gi = hash::pcg(Simd::splat(self.seed) ^ i.cast());

            // Compute the contribution from the gradient
            // n0 = grad0 * (1 - x0^2)^4 * x0
            let x2 = dx * dx;
            let t = Simd::<f32, LANES>::splat(1.0) - x2;
            let t2 = t * t;
            let t4 = t2 * t2;
            let gx = gradient_1d::<LANES>(gi);
            let n = t4 * gx * dx;

            value += n;
            diff_a += t2 * t * gx * x2;
            diff_b += t4 * gx;
        }

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
            value: value * Simd::splat(SCALE),
            derivative: [(diff_a * Simd::splat(-8.0) + diff_b) * Simd::splat(SCALE)],
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
fn gradient_1d<const LANES: usize>(hash: Simd<i32, LANES>) -> Simd<f32, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let h = hash & Simd::splat(0xF);
    let v = (Simd::splat(1) + (h & Simd::splat(7))).cast::<f32>();

    let h_and_8 = (h & Simd::splat(8)).simd_eq(Simd::splat(0));
    h_and_8.select(v, Simd::splat(0.0) - v)
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
    pub fn sample<const LANES: usize>(&self, point: [Simd<f32, LANES>; 2]) -> Sample<LANES, 2>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let (is, xs) = grid::Simplex.get(point);

        let mut value = Simd::splat(0.0);
        let mut diff_a = [Simd::splat(0.0); 2];
        let mut diff_b = [Simd::splat(0.0); 2];

        for v in 0..3 {
            let gi = hash::pcg_3d([is[v][0], is[v][1], Simd::splat(self.seed)])[0];

            // Weights associated with the gradients at each corner
            // These FMA operations are equivalent to: let t = max(0, 0.5 - x*x - y*y)
            let [dx, dy] = xs[v];
            let t = dy
                .mul_add(-dy, dx.mul_add(-dx, Simd::splat(0.5)))
                .simd_max(Simd::splat(0.0));

            let t2 = t * t;
            let t4 = t2 * t2;

            let [gx, gy] = gradient_2d(gi);
            let g = gx * dx + gy * dy;
            let n = t4 * g;
            value += n;

            let temp = t2 * t * g;
            diff_a[0] += temp * dx;
            diff_a[1] += temp * dy;
            diff_b[0] += t4 * gx;
            diff_b[1] += t4 * gy;
        }

        // Scaling factor found by numerical optimization
        const SCALE: f32 = 45.26450774985561631259;
        Sample {
            value: value * Simd::splat(SCALE),
            derivative: std::array::from_fn(|i| {
                (diff_a[i] * Simd::splat(-8.0) + diff_b[i]) * Simd::splat(SCALE)
            }),
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
fn gradient_2d<const LANES: usize>(hash: Simd<i32, LANES>) -> [Simd<f32, LANES>; 2]
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let h = hash & Simd::splat(7);

    let mask = h.simd_lt(Simd::splat(4));
    let x_magnitude = mask.select(Simd::splat(1.0), Simd::splat(2.0));
    let y_magnitude = mask.select(Simd::splat(2.0), Simd::splat(1.0));

    let h_and_1 = (h & Simd::splat(1)).simd_eq(Simd::splat(0));
    let h_and_2 = (h & Simd::splat(2)).simd_eq(Simd::splat(0));

    let gx = mask
        .select_mask(h_and_1, h_and_2)
        .select(x_magnitude, -x_magnitude);
    let gy = mask
        .select_mask(h_and_2, h_and_1)
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
    pub fn sample<const LANES: usize>(&self, point: [Simd<f32, LANES>; 3]) -> Sample<LANES, 3>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let (is, xs) = grid::Simplex.get(point);

        let mut value = Simd::splat(0.0);
        let mut diff_a = [Simd::splat(0.0); 3];
        let mut diff_b = [Simd::splat(0.0); 3];

        for v in 0..4 {
            let [dx, dy, dz] = xs[v];

            // Compute base weight factors associated with each vertex, `0.5 - v . v` where v is the
            // difference between the sample point and the vertex.
            let t = (Simd::splat(0.5) - dx * dx - dy * dy - dz * dz).simd_max(Simd::splat(0.0));

            let t2 = t * t;
            let t4 = t2 * t2;

            // Compute contribution from each vertex
            let g = Gradient3d::new(self.seed, is[v]);
            let gd = g.dot([dx, dy, dz]);
            let v = t4 * gd;

            value += v;

            let temp = t2 * t * gd;
            let offset = [dx, dy, dz];
            let gv = g.vector();
            for i in 0..3 {
                diff_a[i] += temp * offset[i];
                diff_b[i] += t4 * gv[i];
            }
        }

        // Scaling factor found by numerical optimization
        const SCALE: f32 = 67.79816627147162;

        Sample {
            value: value * Simd::splat(SCALE),
            derivative: std::array::from_fn(|i| {
                (diff_a[i] * Simd::splat(-8.0) + diff_b[i]) * Simd::splat(SCALE)
            }),
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
struct Gradient3d<const LANES: usize>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    // Masks guiding dimension selection
    l8: Mask<i32, LANES>,
    l4: Mask<i32, LANES>,
    h12_or_14: Mask<i32, LANES>,

    // Signs for the selected dimensions
    h1: Simd<i32, LANES>,
    h2: Simd<i32, LANES>,
}

impl<const LANES: usize> Gradient3d<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    /// Compute hash values used by `grad3d` and `grad3d_dot`
    #[inline(always)]
    fn new(seed: i32, [i, j, k]: [Simd<i32, LANES>; 3]) -> Self {
        let hash = hash::pcg_4d([i, j, k, Simd::splat(seed)])[0];
        let hasha13 = hash & Simd::splat(13);
        Self {
            l8: hasha13.simd_lt(Simd::splat(8)),
            l4: hasha13.simd_lt(Simd::splat(2)),
            h12_or_14: hasha13.simd_eq(Simd::splat(12)),

            h1: hash << Simd::splat(31),
            h2: (hash & Simd::splat(2)) << Simd::splat(30),
        }
    }

    /// Computes the dot product of a vector with the gradient vector
    #[inline(always)]
    fn dot(&self, [x, y, z]: [Simd<f32, LANES>; 3]) -> Simd<f32, LANES> {
        let u = self.l8.select(x, y);
        let v = self.l4.select(y, self.h12_or_14.select(x, z));
        // Maybe flip sign bits, then sum
        Simd::<f32, LANES>::from_bits(u.to_bits() ^ self.h1.cast())
            + Simd::<f32, LANES>::from_bits(v.to_bits() ^ self.h2.cast())
    }

    /// The gradient vector generated by `dot`
    ///
    /// This is a separate function because it's slower than `grad3d_dot` and only needed when computing
    /// derivatives.
    #[inline(always)]
    fn vector(&self) -> [Simd<f32, LANES>; 3] {
        let first = Simd::<f32, LANES>::from_bits(
            self.h1.cast() | Simd::<f32, LANES>::splat(1.0).to_bits(),
        );
        let gx = self.l8.select(first, Simd::splat(0.0));
        let gy = (!self.l8).select(first, Simd::splat(0.0));

        let second = Simd::<f32, LANES>::from_bits(
            self.h2.cast() | Simd::<f32, LANES>::splat(1.0).to_bits(),
        );
        let gy = self.l4.select(second, gy);
        let gx = (!self.l4 & self.h12_or_14).select(second, gx);
        let gz = (!(self.h12_or_14 | self.l4)).select(second, Simd::splat(0.0));

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
    pub fn sample<const LANES: usize>(&self, point: [Simd<f32, LANES>; 4]) -> Sample<LANES, 4>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let (is, xs) = grid::Simplex.get(point);

        let mut value = Simd::splat(0.0);
        let mut diff_a = [Simd::splat(0.0); 4];
        let mut diff_b = [Simd::splat(0.0); 4];

        for v in 0..5 {
            // Hash the integer coordinates
            let gi = Gradient4d::new(self.seed, is[v]);

            // Compute base weight factors associated with the vertex
            // These FMA operations are equivalent to: let t = max(0, 0.5 - x*x - y*y - z*z - w*w)
            let [dx, dy, dz, dw] = xs[v];
            let t = dw
                .mul_add(
                    -dw,
                    dz.mul_add(-dz, dy.mul_add(-dy, dx.mul_add(-dx, Simd::splat(0.5)))),
                )
                .simd_max(Simd::splat(0.0));

            // Cube weight
            let t2 = t * t;
            let t4 = t2 * t2;

            // Compute gradient's contribution
            let gd = gi.dot(xs[v]);
            let n = t4 * gd;

            value += n;
            let temp = t2 * t * gd;
            let gv = gi.vector();
            for i in 0..4 {
                diff_a[i] += temp * xs[v][i];
                diff_b[i] += t4 * gv[i];
            }
        }

        // Scaling factor found by numerical optimization
        const SCALE: f32 = 62.77772078955791;

        Sample {
            value: value * Simd::splat(SCALE),
            derivative: std::array::from_fn(|i| {
                (diff_a[i] * Simd::splat(-8.0) + diff_b[i]) * Simd::splat(SCALE)
            }),
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
struct Gradient4d<const LANES: usize>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    // Masks guiding dimension selection
    l24: Mask<i32, LANES>,
    l16: Mask<i32, LANES>,
    l8: Mask<i32, LANES>,

    // Signs for the selected dimensions
    sign1: Mask<i32, LANES>,
    sign2: Mask<i32, LANES>,
    sign3: Mask<i32, LANES>,
}

impl<const LANES: usize> Gradient4d<LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    #[inline(always)]
    fn new(seed: i32, [i, j, k, l]: [Simd<i32, LANES>; 4]) -> Self {
        let hash = hash::pcg_4d([
            i ^ Simd::splat(seed),
            j ^ Simd::splat(seed),
            k ^ Simd::splat(seed),
            l ^ Simd::splat(seed),
        ])[0];

        let h = hash & Simd::splat(31);
        Self {
            l24: Simd::splat(24).simd_gt(h),
            l16: Simd::splat(16).simd_gt(h),
            l8: Simd::splat(8).simd_gt(h),

            sign1: Simd::splat(0).simd_eq(h & Simd::splat(1)),
            sign2: Simd::splat(0).simd_eq(h & Simd::splat(2)),
            sign3: Simd::splat(0).simd_eq(h & Simd::splat(4)),
        }
    }

    /// Directly compute the dot product of the gradient vector with a vector
    #[inline(always)]
    fn dot(&self, [x, y, z, t]: [Simd<f32, LANES>; 4]) -> Simd<f32, LANES> {
        let u = self.l24.select(x, y);
        let v = self.l16.select(y, z);
        let w = self.l8.select(z, t);
        self.sign1.select(u, -u) + self.sign2.select(v, -v) + self.sign3.select(w, -w)
    }

    /// Compute the actual gradient vector
    ///
    /// Slower than `dot` and only needed to compute derivatives
    #[inline(always)]
    fn vector(&self) -> [Simd<f32, LANES>; 4] {
        // Select axes
        //       h: u  v  w
        // 24..=31: y, z, t
        // 17..=23: x, z, t
        //  8..=16: x, y, t
        //  0..=7 : x, y, z
        let gx = self.l24.select(Simd::splat(1), Simd::splat(0));
        let gy = (self.l16 | !self.l24).select(Simd::splat(1), Simd::splat(0));
        let gz = (self.l8 | !self.l16).select(Simd::splat(1), Simd::splat(0));
        let gt = (!self.l8).select(Simd::splat(1), Simd::splat(0));

        // Select signs
        let gx = self.sign1.select(gx, -gx);
        let gy = self.l24.select_mask(self.sign2, self.sign1).select(gy, -gy);
        let gz = self.l16.select_mask(self.sign3, self.sign2).select(gz, -gz);
        let gt = self.sign3.select(gt, -gt);

        [gx.cast(), gy.cast(), gz.cast(), gt.cast()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            let n = G.sample::<1>([Simd::splat(x as f32 / 10.0)]).value[0];
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
            let n0 = G.sample::<1>([Simd::splat(center - H)]).value[0];
            let Sample {
                value: n1,
                derivative: d1,
            } = G.sample::<1>([Simd::splat(center)]);
            let n2 = G.sample::<1>([Simd::splat(center + H)]).value[0];
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
                    .sample::<1>([Simd::splat(x as f32 / 10.0), Simd::splat(y as f32 / 10.0)])
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
                } = G.sample::<1>([Simd::splat(center_x), Simd::splat(center_y)]);
                let (value, d) = (value[0], [d[0][0], d[1][0]]);
                let left = G
                    .sample::<1>([Simd::splat(center_x - H), Simd::splat(center_y)])
                    .value[0];
                let right = G
                    .sample::<1>([Simd::splat(center_x + H), Simd::splat(center_y)])
                    .value[0];
                let down = G
                    .sample::<1>([Simd::splat(center_x), Simd::splat(center_y - H)])
                    .value[0];
                let up = G
                    .sample::<1>([Simd::splat(center_x), Simd::splat(center_y + H)])
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
                    let vertex = [Simd::<i32, 1>::splat(i), Simd::splat(j), Simd::splat(k)];
                    let g = Gradient3d::new(0, vertex);
                    let grad = g.vector();
                    assert_eq!(
                        g.dot([Simd::splat(1.0), Simd::splat(10.0), Simd::splat(100.0)])[0],
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
                        .sample::<1>([
                            Simd::splat(x as f32 / 10.0),
                            Simd::splat(y as f32 / 10.0),
                            Simd::splat(z as f32 / 10.0),
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
                    } = G.sample::<1>([
                        Simd::splat(center_x),
                        Simd::splat(center_y),
                        Simd::splat(center_z),
                    ]);
                    let (value, d) = (value[0], [d[0][0], d[1][0], d[2][0]]);
                    let right = G
                        .sample::<1>([
                            Simd::splat(center_x + H),
                            Simd::splat(center_y),
                            Simd::splat(center_z),
                        ])
                        .value[0];
                    let up = G
                        .sample::<1>([
                            Simd::splat(center_x),
                            Simd::splat(center_y + H),
                            Simd::splat(center_z),
                        ])
                        .value[0];
                    let forward = G
                        .sample::<1>([
                            Simd::splat(center_x),
                            Simd::splat(center_y),
                            Simd::splat(center_z + H),
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
                            Simd::<i32, 1>::splat(i),
                            Simd::splat(j),
                            Simd::splat(k),
                            Simd::splat(l),
                        ];
                        let g = Gradient4d::new(0, vertex);
                        let grad = g.vector();
                        assert_eq!(
                            g.dot([
                                Simd::splat(1.0),
                                Simd::splat(10.0),
                                Simd::splat(100.0),
                                Simd::splat(1000.0)
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
                            .sample::<1>([
                                Simd::splat(x as f32 / 10.0),
                                Simd::splat(y as f32 / 10.0),
                                Simd::splat(z as f32 / 10.0),
                                Simd::splat(w as f32 / 10.0),
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
                        } = G.sample::<1>([
                            Simd::splat(center_x),
                            Simd::splat(center_y),
                            Simd::splat(center_z),
                            Simd::splat(center_w),
                        ]);
                        let (value, d) = (value[0], [d[0][0], d[1][0], d[2][0], d[3][0]]);
                        let right = G
                            .sample::<1>([
                                Simd::splat(center_x + H),
                                Simd::splat(center_y),
                                Simd::splat(center_z),
                                Simd::splat(center_w),
                            ])
                            .value[0];
                        let up = G
                            .sample::<1>([
                                Simd::splat(center_x),
                                Simd::splat(center_y + H),
                                Simd::splat(center_z),
                                Simd::splat(center_w),
                            ])
                            .value[0];
                        let forward = G
                            .sample::<1>([
                                Simd::splat(center_x),
                                Simd::splat(center_y),
                                Simd::splat(center_z + H),
                                Simd::splat(center_w),
                            ])
                            .value[0];
                        let fourthward = G
                            .sample::<1>([
                                Simd::splat(center_x),
                                Simd::splat(center_y),
                                Simd::splat(center_z),
                                Simd::splat(center_w + H),
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
