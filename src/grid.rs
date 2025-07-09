use fearless_simd::{Bytes, Select, Simd, SimdBase, SimdCvtFloat, SimdFloat, SimdInt};

/// A distribution of points across a space
pub trait Grid<const DIMENSION: usize> {
    const VERTICES: usize;

    /// Array containing a `T` per dimension per vertex
    type VertexArray<T>: AsRef<[[T; DIMENSION]]>;

    /// Compute integer coordinates of and vectors to `point` from each vertex in the cell enclosing
    /// `point`
    fn get<S: Simd>(
        &self,
        point: [S::f32s; DIMENSION],
    ) -> (Self::VertexArray<S::i32s>, Self::VertexArray<S::f32s>);
}

/// A regular grid of simplices, the simplest possible polytope in a given dimension
pub struct Simplex;

impl Grid<1> for Simplex {
    const VERTICES: usize = 2;

    type VertexArray<T> = [[T; 1]; 2];

    #[inline(always)]
    fn get<S: Simd>(&self, [x]: [S::f32s; 1]) -> ([[S::i32s; 1]; 2], [[S::f32s; 1]; 2]) {
        let i = x.floor();
        let i0 = i.to_int();
        let i1 = i0 + 1;
        let x0 = x - i;
        let x1 = x0 - 1.0;
        ([[i0], [i1]], [[x0], [x1]])
    }
}

impl Grid<2> for Simplex {
    const VERTICES: usize = 3;

    type VertexArray<T> = [[T; 2]; 3];

    #[inline(always)]
    fn get<S: Simd>(&self, [x, y]: [S::f32s; 2]) -> ([[S::i32s; 2]; 3], [[S::f32s; 2]; 3]) {
        let skew = skew_factor(2);
        let unskew = -unskew_factor(2);

        // Skew to distort simplexes with side length sqrt(2)/sqrt(3) until they make up
        // squares
        let s = (x + y) * skew;
        let ips = (x + s).floor();
        let jps = (y + s).floor();

        // Integer coordinates for the base vertex of the triangle
        let i = ips.to_int();
        let j = jps.to_int();

        let t = S::f32s::float_from(i + j) * unskew;

        // Unskewed distances to the first point of the enclosing simplex
        let x0: S::f32s = x - (ips - t);
        let y0: S::f32s = y - (jps - t);

        let i1 = x0.simd_ge(y0).bitcast::<S::i32s>();
        let j1 = y0.simd_gt(x0).bitcast::<S::i32s>();

        // Distances to the second and third points of the enclosing simplex
        let x1 = x0 + S::f32s::float_from(i1) + unskew;
        let y1 = y0 + S::f32s::float_from(j1) + unskew;
        let x2 = x0 - 1.0 + 2.0 * unskew;
        let y2 = y0 - 1.0 + 2.0 * unskew;

        (
            [[i, j], [i - i1, j - j1], [i + 1, j + 1]],
            [[x0, y0], [x1, y1], [x2, y2]],
        )
    }
}

impl Grid<3> for Simplex {
    const VERTICES: usize = 4;

    type VertexArray<T> = [[T; 3]; 4];

    #[inline(always)]
    fn get<S: Simd>(&self, [x, y, z]: [S::f32s; 3]) -> ([[S::i32s; 3]; 4], [[S::f32s; 3]; 4]) {
        let skew = skew_factor(3);
        let unskew = -unskew_factor(3);

        // Find skewed simplex grid coordinates associated with the input coordinates
        let f = (x + y + z) * skew;
        let x0 = (x + f).floor();
        let y0 = (y + f).floor();
        let z0 = (z + f).floor();

        // Integer grid coordinates
        let i = x0.to_int();
        let j = y0.to_int();
        let k = z0.to_int();

        let g = (x0 + y0 + z0) * unskew;
        let x0 = x - (x0 - g);
        let y0 = y - (y0 - g);
        let z0 = z - (z0 - g);

        let x0_ge_y0 = x0.simd_ge(y0);
        let y0_ge_z0 = y0.simd_ge(z0);
        let x0_ge_z0 = x0.simd_ge(z0);

        let i1 = x0_ge_y0 & x0_ge_z0;
        let j1 = !x0_ge_y0 & y0_ge_z0;
        let k1 = !x0_ge_z0 & !y0_ge_z0;

        let i2 = x0_ge_y0 | x0_ge_z0;
        let j2 = !x0_ge_y0 | y0_ge_z0;
        let k2 = !(x0_ge_z0 & y0_ge_z0);

        let v1x = i - i1.bitcast::<S::i32s>();
        let v1y = j - j1.bitcast::<S::i32s>();
        let v1z = k - k1.bitcast::<S::i32s>();

        let v2x = i - i2.bitcast::<S::i32s>();
        let v2y = j - j2.bitcast::<S::i32s>();
        let v2z = k - k2.bitcast::<S::i32s>();

        let v3x = i + 1;
        let v3y = j + 1;
        let v3z = k + 1;

        let x1 = x0 + i1.bitcast::<S::i32s>().to_float::<S::f32s>() + unskew;
        let y1 = y0 + j1.bitcast::<S::i32s>().to_float::<S::f32s>() + unskew;
        let z1 = z0 + k1.bitcast::<S::i32s>().to_float::<S::f32s>() + unskew;

        let x2 = x0 + i2.bitcast::<S::i32s>().to_float::<S::f32s>() + skew;
        let y2 = y0 + j2.bitcast::<S::i32s>().to_float::<S::f32s>() + skew;
        let z2 = z0 + k2.bitcast::<S::i32s>().to_float::<S::f32s>() + skew;

        let x3 = x0 - 0.5;
        let y3 = y0 - 0.5;
        let z3 = z0 - 0.5;

        (
            [[i, j, k], [v1x, v1y, v1z], [v2x, v2y, v2z], [v3x, v3y, v3z]],
            [[x0, y0, z0], [x1, y1, z1], [x2, y2, z2], [x3, y3, z3]],
        )
    }
}

impl Grid<4> for Simplex {
    const VERTICES: usize = 5;

    type VertexArray<T> = [[T; 4]; 5];

    #[inline(always)]
    fn get<S: Simd>(&self, [x, y, z, w]: [S::f32s; 4]) -> ([[S::i32s; 4]; 5], [[S::f32s; 4]; 5]) {
        let skew = skew_factor(4);
        let unskew = -unskew_factor(4);

        let s = (x + y + z + w) * skew;

        let ips = (x + s).floor();
        let jps = (y + s).floor();
        let kps = (z + s).floor();
        let lps = (w + s).floor();

        let i = ips.to_int::<S::i32s>();
        let j = jps.to_int::<S::i32s>();
        let k = kps.to_int::<S::i32s>();
        let l = lps.to_int::<S::i32s>();

        let t = (i + j + k + l).to_float::<S::f32s>() * unskew;
        let x0 = x - (ips - t);
        let y0 = y - (jps - t);
        let z0 = z - (kps - t);
        let w0 = w - (lps - t);

        let mut rank_x = S::i32s::splat(x.witness(), 0);
        let mut rank_y = S::i32s::splat(x.witness(), 0);
        let mut rank_z = S::i32s::splat(x.witness(), 0);
        let mut rank_w = S::i32s::splat(x.witness(), 0);

        let _1 = S::i32s::splat(x.witness(), 1);
        let _0 = S::i32s::splat(x.witness(), 0);
        let cond = x0.simd_gt(y0);
        rank_x += cond.select(_1, _0);
        rank_y += cond.select(_0, _1);
        let cond = x0.simd_gt(z0);
        rank_x += cond.select(_1, _0);
        rank_z += cond.select(_0, _1);
        let cond = x0.simd_gt(w0);
        rank_x += cond.select(_1, _0);
        rank_w += cond.select(_0, _1);
        let cond = y0.simd_gt(z0);
        rank_y += cond.select(_1, _0);
        rank_z += cond.select(_0, _1);
        let cond = y0.simd_gt(w0);
        rank_y += cond.select(_1, _0);
        rank_w += cond.select(_0, _1);
        let cond = z0.simd_gt(w0);
        rank_z += cond.select(_1, _0);
        rank_w += cond.select(_0, _1);

        let _2 = S::i32s::splat(x.witness(), 2);
        let i1 = rank_x.simd_gt(_2).bitcast::<S::i32s>();
        let j1 = rank_y.simd_gt(_2).bitcast::<S::i32s>();
        let k1 = rank_z.simd_gt(_2).bitcast::<S::i32s>();
        let l1 = rank_w.simd_gt(_2).bitcast::<S::i32s>();

        let i2 = rank_x.simd_gt(_1).bitcast::<S::i32s>();
        let j2 = rank_y.simd_gt(_1).bitcast::<S::i32s>();
        let k2 = rank_z.simd_gt(_1).bitcast::<S::i32s>();
        let l2 = rank_w.simd_gt(_1).bitcast::<S::i32s>();

        let i3 = rank_x.simd_gt(_0).bitcast::<S::i32s>();
        let j3 = rank_y.simd_gt(_0).bitcast::<S::i32s>();
        let k3 = rank_z.simd_gt(_0).bitcast::<S::i32s>();
        let l3 = rank_w.simd_gt(_0).bitcast::<S::i32s>();

        let x1 = x0 + i1.to_float::<S::f32s>() + unskew;
        let y1 = y0 + j1.to_float::<S::f32s>() + unskew;
        let z1 = z0 + k1.to_float::<S::f32s>() + unskew;
        let w1 = w0 + l1.to_float::<S::f32s>() + unskew;
        let x2 = x0 + i2.to_float::<S::f32s>() + (2.0 * unskew);
        let y2 = y0 + j2.to_float::<S::f32s>() + (2.0 * unskew);
        let z2 = z0 + k2.to_float::<S::f32s>() + (2.0 * unskew);
        let w2 = w0 + l2.to_float::<S::f32s>() + (2.0 * unskew);
        let x3 = x0 + i3.to_float::<S::f32s>() + (3.0 * unskew);
        let y3 = y0 + j3.to_float::<S::f32s>() + (3.0 * unskew);
        let z3 = z0 + k3.to_float::<S::f32s>() + (3.0 * unskew);
        let w3 = w0 + l3.to_float::<S::f32s>() + (3.0 * unskew);
        let x4 = x0 + (4.0 * unskew - 1.0);
        let y4 = y0 + (4.0 * unskew - 1.0);
        let z4 = z0 + (4.0 * unskew - 1.0);
        let w4 = w0 + (4.0 * unskew - 1.0);

        (
            [
                [i, j, k, l],
                [i - i1, j - j1, k - k1, l - l1],
                [i - i2, j - j2, k - k2, l - l2],
                [i - i3, j - j3, k - k3, l - l3],
                [i, j, k, l].map(|x| x + _1),
            ],
            [
                [x0, y0, z0, w0],
                [x1, y1, z1, w1],
                [x2, y2, z2, w2],
                [x3, y3, z3, w3],
                [x4, y4, z4, w4],
            ],
        )
    }
}

#[inline(always)]
pub fn skew_factor(dimension: usize) -> f32 {
    (((dimension + 1) as f32).sqrt() - 1.0) / dimension as f32
}

#[inline(always)]
pub fn unskew_factor(dimension: usize) -> f32 {
    ((1.0 / ((dimension + 1) as f32).sqrt()) - 1.0) / dimension as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::*;

    #[test]
    fn skew_factors() {
        assert_abs_diff_eq!(skew_factor(2), (3.0f32.sqrt() - 1.0) / 2.0);
        assert_abs_diff_eq!(skew_factor(3), 1.0 / 3.0);
        assert_abs_diff_eq!(skew_factor(4), (5.0f32.sqrt() - 1.0) / 4.0);

        assert_abs_diff_eq!(unskew_factor(2), -(3.0 - 3.0f32.sqrt()) / 6.0);
        assert_abs_diff_eq!(unskew_factor(3), -1.0 / 6.0);
        assert_abs_diff_eq!(unskew_factor(4), -(5.0 - 5.0f32.sqrt()) / 20.0);
    }
}
