use fearless_simd::Simd;

#[inline(always)]
pub fn pcg<S: Simd>(v: S::u32s) -> S::u32s {
    // PCG hash function from "Hash Functions for GPU Rendering"
    let state = v * 747796405 + 2891336453u32;
    let word = ((state >> ((state >> 28) + 4)) ^ state) * 277803737;
    (word >> 22) ^ word
}

// For completeness
#[allow(dead_code)]
#[inline(always)]
pub fn pcg_2d<S: Simd>([mut vx, mut vy]: [S::u32s; 2]) -> [S::u32s; 2] {
    vx = vx * 1664525 + 1013904223;
    vy = vy * 1664525 + 1013904223;

    vx += vy * 1664525;
    vy += vx * 1664525;

    vx ^= vx >> 16;
    vy ^= vy >> 16;

    vx += vy * 1664525;
    vy += vx * 1664525;

    vx ^= vx >> 16;
    vy ^= vy >> 16;

    [vx, vy]
}

#[inline(always)]
pub fn pcg_3d<S: Simd>([mut vx, mut vy, mut vz]: [S::u32s; 3]) -> [S::u32s; 3] {
    // PCG3D hash function from "Hash Functions for GPU Rendering"
    vx = vx * 1664525 + 1013904223;
    vy = vy * 1664525 + 1013904223;
    vz = vz * 1664525 + 1013904223;

    vx += vy * vz;
    vy += vz * vx;
    vz += vx * vy;

    vx = vx ^ (vx >> 16);
    vy = vy ^ (vy >> 16);
    vz = vz ^ (vz >> 16);

    vx += vy * vz;
    vy += vz * vx;
    vz += vx * vy;

    [vx, vy, vz]
}

#[inline(always)]
pub fn pcg_4d<S: Simd>([mut vx, mut vy, mut vz, mut vw]: [S::u32s; 4]) -> [S::u32s; 4] {
    // PCG4D hash function from "Hash Functions for GPU Rendering"
    vx = vx * 1664525 + 1013904223;
    vy = vy * 1664525 + 1013904223;
    vz = vz * 1664525 + 1013904223;
    vw = vw * 1664525 + 1013904223;

    vx += vy * vw;
    vy += vz * vx;
    vz += vx * vy;
    vw += vy * vz;

    vx = vx ^ (vx >> 16);
    vy = vy ^ (vy >> 16);
    vz = vz ^ (vz >> 16);
    vw = vw ^ (vw >> 16);

    vx += vy * vw;
    vy += vz * vx;
    vz += vx * vy;
    vw += vy * vz;

    [vx, vy, vz, vw]
}
