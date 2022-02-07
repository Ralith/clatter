#![feature(portable_simd)]

use std::simd::{LaneCount, Simd, StdFloat as _, SupportedLaneCount};

#[cfg(feature = "rand")]
use rand::{
    distributions::{Distribution, Standard},
    seq::SliceRandom,
    Rng,
};

#[derive(Debug, Clone)]
pub struct Generator {
    perm: [i32; 512],
}

impl Generator {
    pub const fn new() -> Self {
        Self { perm: DEFAULT_PERM }
    }

    #[cfg(feature = "rand")]
    pub fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let mut perm = [0; 512];
        for i in 0..256 {
            perm[i] = i as i32;
        }
        perm[..256].shuffle(rng);
        let (left, right) = perm.split_at_mut(256);
        right.copy_from_slice(left);
        Self { perm }
    }

    pub fn simplex_1d<const LANES: usize>(&self, [x]: [Simd<f32, LANES>; 1]) -> Sample<LANES, 1>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        // Gradients are selected deterministically based on the whole part of `x`
        let ips = x.floor();
        let i0 = ips.cast::<i32>();
        let i1 = (i0 + Simd::splat(1)) & Simd::splat(0xFF);

        // the fractional part of x, i.e. the distance to the left gradient node. 0 ≤ x0 < 1.
        let x0 = x - ips;
        // signed distance to the right gradient node
        let x1 = x0 - Simd::splat(1.0);

        // Select gradients
        let i0 = i0 & Simd::splat(0xFF);
        let gi0 = Simd::<i32, LANES>::gather_or_default(&self.perm, i0.cast());
        let gi1 = Simd::<i32, LANES>::gather_or_default(&self.perm, i1.cast());

        // Compute the contribution from the first gradient
        // n0 = grad0 * (1 - x0^2)^4 * x0
        let x20 = x0 * x0;
        let t0 = Simd::<f32, LANES>::splat(1.0) - x20;
        let t20 = t0 * t0;
        let t40 = t20 * t20;
        let gx0 = gradient_1d::<LANES>(gi0);
        let n0 = t40 * gx0 * x0;

        // Compute the contribution from the second gradient
        // n1 = grad1 * (x0 - 1) * (1 - (x0 - 1)^2)^4
        let x21 = x1 * x1;
        let t1 = Simd::<f32, LANES>::splat(1.0) - x21;
        let t21 = t1 * t1;
        let t41 = t21 * t21;
        let gx1 = gradient_1d::<LANES>(gi1);
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
        const SCALE: f32 = 256.0 / (81.0 * 7.0);
        Sample {
            value: (n0 + n1) * Simd::splat(SCALE),
            derivative: [
                ((t20 * t0 * gx0 * x20 + t21 * t1 * gx1 * x21) * Simd::splat(-8.0)
                    + t40 * gx0
                    + t41 * gx1)
                    * Simd::splat(SCALE),
            ],
        }
    }

    pub fn simplex_2d<const LANES: usize>(&self, [x, y]: [Simd<f32, LANES>; 2]) -> Sample<LANES, 2>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        const SKEW: f32 = 0.36602540378;
        const UNSKEW: f32 = 0.2113248654;

        // Skew to distort simplexes with side length sqrt(2)/sqrt(3) until they make up
        // squares
        let s = (x + y) * Simd::splat(SKEW);
        let ips = (x + s).floor();
        let jps = (y + s).floor();

        // Integer coordinates for the base vertex of the triangle
        let i = ips.cast::<i32>();
        let j = jps.cast::<i32>();

        let t = (i + j).cast::<f32>() * Simd::splat(UNSKEW);

        // Unskewed distances to the first point of the enclosing simplex
        let x0 = x - (ips - t);
        let y0 = y - (jps - t);

        let i1 = x0.lanes_ge(y0).to_int();
        let j1 = y0.lanes_gt(x0).to_int();

        // Distances to the second and third points of the enclosing simplex
        let x1 = x0 + i1.cast() + Simd::splat(UNSKEW);
        let y1 = y0 + j1.cast() + Simd::splat(UNSKEW);
        let x2 = x0 + Simd::splat(-1.0) + Simd::splat(2.0 * UNSKEW);
        let y2 = y0 + Simd::splat(-1.0) + Simd::splat(2.0 * UNSKEW);

        let ii = i & Simd::splat(0xFF);
        let jj = j & Simd::splat(0xFF);

        let gi0 = Simd::<i32, LANES>::gather_or_default(
            &self.perm,
            (ii + Simd::gather_or_default(&self.perm, jj.cast())).cast(),
        );

        let gi1 = Simd::<i32, LANES>::gather_or_default(
            &self.perm,
            ((ii - i1) + Simd::gather_or_default(&self.perm, (jj - j1).cast())).cast(),
        );

        let gi2 = Simd::<i32, LANES>::gather_or_default(
            &self.perm,
            ((ii - Simd::splat(-1))
                + Simd::gather_or_default(&self.perm, (jj - Simd::splat(-1)).cast()))
            .cast(),
        );

        // Weights associated with the gradients at each corner
        // These FMA operations are equivalent to: let t = max(0, 0.5 - x*x - y*y)
        let t0 = y0
            .mul_add(-y0, x0.mul_add(-x0, Simd::splat(0.5)))
            .max(Simd::splat(0.0));
        let t1 = y1
            .mul_add(-y1, x1.mul_add(-x1, Simd::splat(0.5)))
            .max(Simd::splat(0.0));
        let t2 = y2
            .mul_add(-y2, x2.mul_add(-x2, Simd::splat(0.5)))
            .max(Simd::splat(0.0));

        let t20 = t0 * t0;
        let t40 = t20 * t20;
        let t21 = t1 * t1;
        let t41 = t21 * t21;
        let t22 = t2 * t2;
        let t42 = t22 * t22;

        let [gx0, gy0] = gradient_2d(gi0);
        let g0 = gx0 * x0 + gy0 * y0;
        let n0 = t40 * g0;
        let [gx1, gy1] = gradient_2d(gi1);
        let g1 = gx1 * x1 + gy1 * y1;
        let n1 = t41 * g1;
        let [gx2, gy2] = gradient_2d(gi2);
        let g2 = gx2 * x2 + gy2 * y2;
        let n2 = t42 * g2;

        // Scaling factor found by numerical optimization
        const SCALE: f32 = 45.26450774985561631259;

        Sample {
            value: (n0 + n1 + n2) * Simd::splat(SCALE),
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
                dnoise_dx *= Simd::splat(-8.0);
                dnoise_dy *= Simd::splat(-8.0);
                dnoise_dx += t40 * gx0 + t41 * gx1 + t42 * gx2;
                dnoise_dy += t40 * gy0 + t41 * gy1 + t42 * gy2;
                dnoise_dx *= Simd::splat(SCALE);
                dnoise_dy *= Simd::splat(SCALE);
                [dnoise_dx, dnoise_dy]
            },
        }
    }
}

#[cfg(feature = "rand")]
impl Distribution<Generator> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Generator {
        Generator::new(rng)
    }
}

impl Default for Generator {
    fn default() -> Self {
        Generator::new()
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Sample<const LANES: usize, const DIMENSION: usize>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    pub value: Simd<f32, LANES>,
    pub derivative: [Simd<f32, LANES>; DIMENSION],
}

const DEFAULT_PERM: [i32; 512] = [
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69,
    142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219,
    203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175,
    74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230,
    220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76,
    132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173,
    186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206,
    59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152, 2, 44, 154, 163,
    70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232,
    178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162,
    241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204,
    176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141,
    128, 195, 78, 66, 215, 61, 156, 180, 151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194,
    233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234,
    75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174,
    20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83,
    111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25,
    63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188,
    159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147,
    118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170,
    213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253,
    19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193,
    238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31,
    181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205, 93,
    222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
];

/// Generates a random integer gradient in ±7 inclusive
///
/// This differs from Gustavson's well-known implementation in that gradients can be zero, and the
/// maximum gradient is 7 rather than 8.
fn gradient_1d<const LANES: usize>(hash: Simd<i32, LANES>) -> Simd<f32, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let h = hash & Simd::splat(0xF);
    let v = (h & Simd::splat(7)).cast::<f32>();

    let h_and_8 = (h & Simd::splat(8)).lanes_eq(Simd::splat(0));
    h_and_8.select(v, Simd::splat(0.0) - v)
}

fn gradient_2d<const LANES: usize>(hash: Simd<i32, LANES>) -> [Simd<f32, LANES>; 2]
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let h = hash & Simd::splat(7);

    let mask = h.lanes_lt(Simd::splat(4));
    let x_magnitude = mask.select(Simd::splat(1.0), Simd::splat(2.0));
    let y_magnitude = mask.select(Simd::splat(2.0), Simd::splat(1.0));

    let h_and_1 = (h & Simd::splat(1)).lanes_eq(Simd::splat(0));
    let h_and_2 = (h & Simd::splat(2)).lanes_eq(Simd::splat(0));

    let gx = mask
        .select_mask(h_and_1, h_and_2)
        .select(x_magnitude, -x_magnitude);
    let gy = mask
        .select_mask(h_and_2, h_and_1)
        .select(y_magnitude, -y_magnitude);
    [gx, gy]
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: Generator = Generator::new();

    fn check_bounds(min: f32, max: f32) {
        dbg!(min, max);
        assert!(min < -0.75 && min >= -1.0, "min out of range {}", min);
        assert!(max > 0.75 && max <= 1.0, "max out of range: {}", max);
    }

    #[test]
    fn simplex_1d_range() {
        let mut min = f32::INFINITY;
        let mut max = -f32::INFINITY;
        for x in 0..1000 {
            let n = G.simplex_1d::<1>([Simd::splat(x as f32 / 10.0)]).value[0];
            min = min.min(n);
            max = max.max(n);
        }
        check_bounds(min, max);
    }

    #[test]
    fn simplex_1d_deriv_sanity() {
        let mut avg_err = 0.0;
        const SEEDS: i32 = 10;
        const POINTS: i32 = 1000;
        for x in 0..POINTS {
            // Offset a bit so we don't check derivative at lattice points, where it's always zero
            let center = x as f32 / 10.0 + 0.1234;
            const H: f32 = 0.01;
            let n0 = G.simplex_1d::<1>([Simd::splat(center - H)]).value[0];
            let Sample {
                value: n1,
                derivative: d1,
            } = G.simplex_1d::<1>([Simd::splat(center)]);
            let n2 = G.simplex_1d::<1>([Simd::splat(center + H)]).value[0];
            let (n1, d1) = (n1[0], d1[0][0]);
            avg_err += ((n2 - (n1 + d1 * H)).abs() + (n0 - (n1 - d1 * H)).abs())
                / (SEEDS * POINTS * 2) as f32;
        }
        assert!(avg_err < 1e-3);
    }

    #[test]
    fn simplex_2d_range() {
        let mut min = f32::INFINITY;
        let mut max = -f32::INFINITY;
        for y in 0..10 {
            for x in 0..100 {
                let n = G
                    .simplex_2d::<1>([Simd::splat(x as f32 / 10.0), Simd::splat(y as f32 / 10.0)])
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
        let mut avg_err = 0.0;
        const SEEDS: i32 = 10;
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
                } = G.simplex_2d::<1>([Simd::splat(center_x), Simd::splat(center_y)]);
                let (value, d) = (value[0], [d[0][0], d[1][0]]);
                let left = G
                    .simplex_2d::<1>([Simd::splat(center_x - H), Simd::splat(center_y)])
                    .value[0];
                let right = G
                    .simplex_2d::<1>([Simd::splat(center_x + H), Simd::splat(center_y)])
                    .value[0];
                let down = G
                    .simplex_2d::<1>([Simd::splat(center_x), Simd::splat(center_y - H)])
                    .value[0];
                let up = G
                    .simplex_2d::<1>([Simd::splat(center_x), Simd::splat(center_y + H)])
                    .value[0];
                avg_err += ((left - (value - d[0] * H)).abs()
                    + (right - (value + d[0] * H)).abs()
                    + (down - (value - d[1] * H)).abs()
                    + (up - (value + d[1] * H)).abs())
                    / (SEEDS * POINTS * POINTS * 4) as f32;
            }
        }
        assert!(avg_err < 1e-3);
    }
}
