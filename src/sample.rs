use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign};

use fearless_simd::{Simd, SimdBase};

#[derive(Debug, Copy, Clone)]
pub struct Sample<S: Simd, const DIMENSION: usize> {
    pub value: S::f32s,
    pub derivative: [S::f32s; DIMENSION],
}

impl<S: Simd, const DIMENSION: usize> Sample<S, DIMENSION> {
    pub fn constant(value: S::f32s) -> Self {
        Self {
            value,
            derivative: [S::f32s::splat(value.witness(), 0.0); DIMENSION],
        }
    }
}

impl<S: Simd, const DIMENSION: usize> AddAssign<Self> for Sample<S, DIMENSION> {
    fn add_assign(&mut self, other: Self) {
        self.value += other.value;
        for i in 0..DIMENSION {
            self.derivative[i] += other.derivative[i];
        }
    }
}

impl<S: Simd, const DIMENSION: usize> Add<Self> for Sample<S, DIMENSION> {
    type Output = Self;
    fn add(mut self, other: Self) -> Self {
        self += other;
        self
    }
}

impl<S: Simd, const DIMENSION: usize> MulAssign<S::f32s> for Sample<S, DIMENSION> {
    fn mul_assign(&mut self, other: S::f32s) {
        self.value *= other;
        for i in 0..DIMENSION {
            self.derivative[i] *= other;
        }
    }
}

impl<S: Simd, const DIMENSION: usize> Mul<S::f32s> for Sample<S, DIMENSION> {
    type Output = Self;
    fn mul(mut self, other: S::f32s) -> Self {
        self *= other;
        self
    }
}

impl<S: Simd, const DIMENSION: usize> DivAssign<S::f32s> for Sample<S, DIMENSION> {
    fn div_assign(&mut self, other: S::f32s) {
        self.value /= other;
        for i in 0..DIMENSION {
            self.derivative[i] /= other;
        }
    }
}

impl<S: Simd, const DIMENSION: usize> Div<S::f32s> for Sample<S, DIMENSION> {
    type Output = Self;
    fn div(mut self, other: S::f32s) -> Self {
        self /= other;
        self
    }
}
