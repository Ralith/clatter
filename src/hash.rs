use fearless_simd::Simd;

#[inline(always)]
pub fn esgtsa<S: Simd>(mut v: S::u32s) -> S::u32s {
    // ESGTSA hash from "Hash Functions for GPU Rendering", detailed in the supplementary paper.
    //
    // The paper recommends the 1D PCG hash, which it says is slightly faster and fails very slightly fewer BigCrush
    // tests. However, it contains a variable right shift, which is slow on AVX2 and scalarized entirely in SSE4.2 and
    // below. In practice, ESGTSA is faster on CPU.
    v = (v ^ 2747636419) * 2654435769;
    v = (v ^ (v >> 16)) * 2654435769;
    v = (v ^ (v >> 16)) * 2654435769;
    v
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
