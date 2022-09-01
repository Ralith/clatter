use std::{
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign},
    simd::{LaneCount, Simd, SupportedLaneCount},
};

#[derive(Debug, Copy, Clone)]
pub struct Sample<const LANES: usize, const DIMENSION: usize>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    pub value: Simd<f32, LANES>,
    pub derivative: [Simd<f32, LANES>; DIMENSION],
}

impl<const LANES: usize, const DIMENSION: usize> Sample<LANES, DIMENSION>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    pub fn constant(value: Simd<f32, LANES>) -> Self {
        Self {
            value,
            derivative: [Simd::splat(0.0); DIMENSION],
        }
    }
}

impl<const LANES: usize, const DIMENSION: usize> Default for Sample<LANES, DIMENSION>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    fn default() -> Self {
        Self::constant(Simd::splat(0.0))
    }
}

impl<const LANES: usize, const DIMENSION: usize> AddAssign<Self> for Sample<LANES, DIMENSION>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    fn add_assign(&mut self, other: Self) {
        self.value += other.value;
        for i in 0..DIMENSION {
            self.derivative[i] += other.derivative[i];
        }
    }
}

impl<const LANES: usize, const DIMENSION: usize> Add<Self> for Sample<LANES, DIMENSION>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;
    fn add(mut self, other: Self) -> Self {
        self += other;
        self
    }
}

impl<const LANES: usize, const DIMENSION: usize> MulAssign<Simd<f32, LANES>>
    for Sample<LANES, DIMENSION>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    fn mul_assign(&mut self, other: Simd<f32, LANES>) {
        self.value *= other;
        for i in 0..DIMENSION {
            self.derivative[i] *= other;
        }
    }
}

impl<const LANES: usize, const DIMENSION: usize> Mul<Simd<f32, LANES>> for Sample<LANES, DIMENSION>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;
    fn mul(mut self, other: Simd<f32, LANES>) -> Self {
        self *= other;
        self
    }
}

impl<const LANES: usize, const DIMENSION: usize> DivAssign<Simd<f32, LANES>>
    for Sample<LANES, DIMENSION>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    fn div_assign(&mut self, other: Simd<f32, LANES>) {
        self.value /= other;
        for i in 0..DIMENSION {
            self.derivative[i] /= other;
        }
    }
}

impl<const LANES: usize, const DIMENSION: usize> Div<Simd<f32, LANES>> for Sample<LANES, DIMENSION>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    type Output = Self;
    fn div(mut self, other: Simd<f32, LANES>) -> Self {
        self /= other;
        self
    }
}
