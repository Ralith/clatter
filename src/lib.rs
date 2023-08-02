#![feature(portable_simd)]

mod grid;
mod hash;
mod sample;
mod simplex;

use grid::Grid;
pub use sample::Sample;
pub use simplex::{Simplex1d, Simplex2d, Simplex3d, Simplex4d};
