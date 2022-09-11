#![feature(portable_simd, generic_associated_types)]

mod grid;
mod hash;
mod sample;
mod simplex;

use grid::Grid;
pub use sample::Sample;
pub use simplex::{Simplex1d, Simplex2d, Simplex3d, Simplex4d};
