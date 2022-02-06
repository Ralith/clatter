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

    pub fn simplex_1d<const LANES: usize>(&self, x: Simd<f32, LANES>) -> Sample<LANES>
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
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
            derivative: ((t20 * t0 * gx0 * x20 + t21 * t1 * gx1 * x21) * Simd::splat(-8.0)
                + t40 * gx0
                + t41 * gx1)
                * Simd::splat(SCALE),
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
pub struct Sample<const LANES: usize>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    pub value: Simd<f32, LANES>,
    pub derivative: Simd<f32, LANES>,
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
            let n = G.simplex_1d::<1>(Simd::splat(x as f32 / 10.0)).value[0];
            min = min.min(n);
            max = max.max(n);
        }
        check_bounds(min, max);
    }
}
