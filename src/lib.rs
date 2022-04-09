#![feature(portable_simd)]

mod hash;
mod sample;
mod simplex;

pub use sample::Sample;
pub use simplex::{Simplex1d, Simplex2d, Simplex3d, Simplex4d};
