use std::simd::{LaneCount, Mask, Simd, SimdPartialOrd, StdFloat as _, SupportedLaneCount};

/// A distribution of points across a space
pub trait Grid<const DIMENSION: usize> {
    const VERTICES: usize;

    /// Array containing a `T` per dimension per vertex
    type VertexArray<T>: AsRef<[[T; DIMENSION]]>;

    /// Compute integer coordinates of and vectors to `point` from each vertex in the cell enclosing
    /// `point`
    fn get<const LANES: usize>(
        &self,
        point: [Simd<f32, LANES>; DIMENSION],
    ) -> (
        Self::VertexArray<Simd<i32, LANES>>,
        Self::VertexArray<Simd<f32, LANES>>,
    )
    where
        LaneCount<LANES>: SupportedLaneCount;
}

/// A regular grid of simplices, the simplest possible polytope in a given dimension
pub struct Simplex;

impl Grid<1> for Simplex {
    const VERTICES: usize = 2;

    type VertexArray<T> = [[T; 1]; 2];

    #[inline(always)]
    fn get<const LANES: usize>(
        &self,
        [x]: [Simd<f32, LANES>; 1],
    ) -> ([[Simd<i32, LANES>; 1]; 2], [[Simd<f32, LANES>; 1]; 2])
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let i = x.floor();
        let i0 = i.cast();
        let i1 = i0 + Simd::splat(1);
        let x0 = x - i;
        let x1 = x0 - Simd::splat(1.0);
        ([[i0], [i1]], [[x0], [x1]])
    }
}

impl Grid<2> for Simplex {
    const VERTICES: usize = 3;

    type VertexArray<T> = [[T; 2]; 3];

    #[inline(always)]
    fn get<const LANES: usize>(
        &self,
        [x, y]: [Simd<f32, LANES>; 2],
    ) -> ([[Simd<i32, LANES>; 2]; 3], [[Simd<f32, LANES>; 2]; 3])
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let skew = skew_factor(2);
        let unskew = -unskew_factor(2);

        // Skew to distort simplexes with side length sqrt(2)/sqrt(3) until they make up
        // squares
        let s = (x + y) * Simd::splat(skew);
        let ips = (x + s).floor();
        let jps = (y + s).floor();

        // Integer coordinates for the base vertex of the triangle
        let i = ips.cast::<i32>();
        let j = jps.cast::<i32>();

        let t = (i + j).cast::<f32>() * Simd::splat(unskew);

        // Unskewed distances to the first point of the enclosing simplex
        let x0 = x - (ips - t);
        let y0 = y - (jps - t);

        let i1 = x0.simd_ge(y0).to_int();
        let j1 = y0.simd_gt(x0).to_int();

        // Distances to the second and third points of the enclosing simplex
        let x1 = x0 + i1.cast() + Simd::splat(unskew);
        let y1 = y0 + j1.cast() + Simd::splat(unskew);
        let x2 = x0 + Simd::splat(-1.0) + Simd::splat(2.0 * unskew);
        let y2 = y0 + Simd::splat(-1.0) + Simd::splat(2.0 * unskew);

        (
            [
                [i, j],
                [i - i1, j - j1],
                [i + Simd::splat(1), j + Simd::splat(1)],
            ],
            [[x0, y0], [x1, y1], [x2, y2]],
        )
    }
}

impl Grid<3> for Simplex {
    const VERTICES: usize = 4;

    type VertexArray<T> = [[T; 3]; 4];

    #[inline(always)]
    fn get<const LANES: usize>(
        &self,
        [x, y, z]: [Simd<f32, LANES>; 3],
    ) -> ([[Simd<i32, LANES>; 3]; 4], [[Simd<f32, LANES>; 3]; 4])
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let skew = skew_factor(3);
        let unskew = -unskew_factor(3);

        const X_PRIME: i32 = 1619;
        const Y_PRIME: i32 = 31337;
        const Z_PRIME: i32 = 6791;

        // Find skewed simplex grid coordinates associated with the input coordinates
        let f = (x + y + z) * Simd::splat(skew);
        let x0 = (x + f).floor();
        let y0 = (y + f).floor();
        let z0 = (z + f).floor();

        // Integer grid coordinates
        let i = x0.cast::<i32>() * Simd::splat(X_PRIME);
        let j = y0.cast::<i32>() * Simd::splat(Y_PRIME);
        let k = z0.cast::<i32>() * Simd::splat(Z_PRIME);

        let g = Simd::splat(unskew) * (x0 + y0 + z0);
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

        let v1x = i + i1.select(Simd::splat(X_PRIME), Simd::splat(0));
        let v1y = j + j1.select(Simd::splat(Y_PRIME), Simd::splat(0));
        let v1z = k + k1.select(Simd::splat(Z_PRIME), Simd::splat(0));

        let v2x = i + i2.select(Simd::splat(X_PRIME), Simd::splat(0));
        let v2y = j + j2.select(Simd::splat(Y_PRIME), Simd::splat(0));
        let v2z = k + k2.select(Simd::splat(Z_PRIME), Simd::splat(0));

        let v3x = i + Simd::splat(X_PRIME);
        let v3y = j + Simd::splat(Y_PRIME);
        let v3z = k + Simd::splat(Z_PRIME);

        let x1 = x0 - i1.select(Simd::splat(1.0), Simd::splat(0.0)) + Simd::splat(unskew);
        let y1 = y0 - j1.select(Simd::splat(1.0), Simd::splat(0.0)) + Simd::splat(unskew);
        let z1 = z0 - k1.select(Simd::splat(1.0), Simd::splat(0.0)) + Simd::splat(unskew);

        let x2 = x0 - i2.select(Simd::splat(1.0), Simd::splat(0.0)) + Simd::splat(skew);
        let y2 = y0 - j2.select(Simd::splat(1.0), Simd::splat(0.0)) + Simd::splat(skew);
        let z2 = z0 - k2.select(Simd::splat(1.0), Simd::splat(0.0)) + Simd::splat(skew);

        let x3 = x0 + Simd::splat(-0.5);
        let y3 = y0 + Simd::splat(-0.5);
        let z3 = z0 + Simd::splat(-0.5);

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
    fn get<const LANES: usize>(
        &self,
        [x, y, z, w]: [Simd<f32, LANES>; 4],
    ) -> ([[Simd<i32, LANES>; 4]; 5], [[Simd<f32, LANES>; 4]; 5])
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let skew = skew_factor(4);
        let unskew = -unskew_factor(4);

        let s = Simd::splat(skew) * (x + y + z + w);

        let ips = (x + s).floor();
        let jps = (y + s).floor();
        let kps = (z + s).floor();
        let lps = (w + s).floor();

        let i = ips.cast::<i32>();
        let j = jps.cast::<i32>();
        let k = kps.cast::<i32>();
        let l = lps.cast::<i32>();

        let t = Simd::splat(unskew) * (i + j + k + l).cast();
        let x0 = x - (ips - t);
        let y0 = y - (jps - t);
        let z0 = z - (kps - t);
        let w0 = w - (lps - t);

        let mut rank_x = Simd::splat(0);
        let mut rank_y = Simd::splat(0);
        let mut rank_z = Simd::splat(0);
        let mut rank_w = Simd::splat(0);

        let cond = x0.simd_gt(y0);
        rank_x += cond.select(Simd::splat(1), Simd::splat(0));
        rank_y += cond.select(Simd::splat(0), Simd::splat(1));
        let cond = x0.simd_gt(z0);
        rank_x += cond.select(Simd::splat(1), Simd::splat(0));
        rank_z += cond.select(Simd::splat(0), Simd::splat(1));
        let cond = x0.simd_gt(w0);
        rank_x += cond.select(Simd::splat(1), Simd::splat(0));
        rank_w += cond.select(Simd::splat(0), Simd::splat(1));
        let cond = y0.simd_gt(z0);
        rank_y += cond.select(Simd::splat(1), Simd::splat(0));
        rank_z += cond.select(Simd::splat(0), Simd::splat(1));
        let cond = y0.simd_gt(w0);
        rank_y += cond.select(Simd::splat(1), Simd::splat(0));
        rank_w += cond.select(Simd::splat(0), Simd::splat(1));
        let cond = z0.simd_gt(w0);
        rank_z += cond.select(Simd::splat(1), Simd::splat(0));
        rank_w += cond.select(Simd::splat(0), Simd::splat(1));

        let i1 = (rank_x.simd_gt(Simd::splat(2)) as Mask<i32, LANES>).to_int();
        let j1 = (rank_y.simd_gt(Simd::splat(2)) as Mask<i32, LANES>).to_int();
        let k1 = (rank_z.simd_gt(Simd::splat(2)) as Mask<i32, LANES>).to_int();
        let l1 = (rank_w.simd_gt(Simd::splat(2)) as Mask<i32, LANES>).to_int();

        let i2 = (rank_x.simd_gt(Simd::splat(1)) as Mask<i32, LANES>).to_int();
        let j2 = (rank_y.simd_gt(Simd::splat(1)) as Mask<i32, LANES>).to_int();
        let k2 = (rank_z.simd_gt(Simd::splat(1)) as Mask<i32, LANES>).to_int();
        let l2 = (rank_w.simd_gt(Simd::splat(1)) as Mask<i32, LANES>).to_int();

        let i3 = (rank_x.simd_gt(Simd::splat(0)) as Mask<i32, LANES>).to_int();
        let j3 = (rank_y.simd_gt(Simd::splat(0)) as Mask<i32, LANES>).to_int();
        let k3 = (rank_z.simd_gt(Simd::splat(0)) as Mask<i32, LANES>).to_int();
        let l3 = (rank_w.simd_gt(Simd::splat(0)) as Mask<i32, LANES>).to_int();

        let x1 = x0 + i1.cast() + Simd::splat(unskew);
        let y1 = y0 + j1.cast() + Simd::splat(unskew);
        let z1 = z0 + k1.cast() + Simd::splat(unskew);
        let w1 = w0 + l1.cast() + Simd::splat(unskew);
        let x2 = x0 + i2.cast() + Simd::splat(2.0 * unskew);
        let y2 = y0 + j2.cast() + Simd::splat(2.0 * unskew);
        let z2 = z0 + k2.cast() + Simd::splat(2.0 * unskew);
        let w2 = w0 + l2.cast() + Simd::splat(2.0 * unskew);
        let x3 = x0 + i3.cast() + Simd::splat(3.0 * unskew);
        let y3 = y0 + j3.cast() + Simd::splat(3.0 * unskew);
        let z3 = z0 + k3.cast() + Simd::splat(3.0 * unskew);
        let w3 = w0 + l3.cast() + Simd::splat(3.0 * unskew);
        let x4 = (x0 - Simd::splat(1.0)) + Simd::splat(4.0 * unskew);
        let y4 = (y0 - Simd::splat(1.0)) + Simd::splat(4.0 * unskew);
        let z4 = (z0 - Simd::splat(1.0)) + Simd::splat(4.0 * unskew);
        let w4 = (w0 - Simd::splat(1.0)) + Simd::splat(4.0 * unskew);

        (
            [
                [i, j, k, l],
                [i - i1, j - j1, k - k1, l - l1],
                [i - i2, j - j2, k - k2, l - l2],
                [i - i3, j - j3, k - k3, l - l3],
                [i, j, k, l].map(|x| x + Simd::splat(1)),
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

/// A regular grid of n-dimensional cubes
pub struct Square;

macro_rules! impl_square {
    ($dim:literal) => {
        impl Grid<$dim> for Square {
            const VERTICES: usize = 2usize.pow($dim);

            type VertexArray<T> = [[T; $dim]; 2usize.pow($dim)];

            #[inline(always)]
            fn get<const LANES: usize>(
                &self,
                point: [Simd<f32, LANES>; $dim],
            ) -> (
                Self::VertexArray<Simd<i32, LANES>>,
                Self::VertexArray<Simd<f32, LANES>>,
            )
            where
                LaneCount<LANES>: SupportedLaneCount,
            {
                const DIMENSION: usize = $dim;
                let base = point.map(|x| x.floor().cast());
                let vertices = std::array::from_fn(|i| {
                    let mut coords = base;
                    for d in 0..DIMENSION {
                        if i & (1 << d) != 0 {
                            coords[d] += Simd::splat(1);
                        }
                    }
                    coords
                });

                let distances = vertices.map(|v| {
                    let mut point = point;
                    for d in 0..DIMENSION {
                        point[d] -= v[d].cast();
                    }
                    point
                });

                (vertices, distances)
            }
        }
    };
}

impl_square!(1);
impl_square!(2);
impl_square!(3);
impl_square!(4);

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

    #[test]
    fn square_smoke() {
        let (vertices, vectors) =
            <Square as Grid<2>>::get::<1>(&Square, [1.25, 4.75].map(Simd::splat));
        assert_eq!(vertices.len(), 4);
        for (vertex, vector) in [
            ([1, 4], [0.25, 0.75]),
            ([1, 5], [0.25, -0.25]),
            ([2, 4], [-0.75, 0.75]),
            ([2, 5], [-0.75, -0.25]),
        ] {
            let i = vertices
                .iter()
                .position(|&x| x == vertex.map(Simd::splat))
                .expect("missing expected vertex");
            for d in 0..2 {
                assert_abs_diff_eq!(vectors[i][d][0], vector[d]);
            }
        }
    }
}
