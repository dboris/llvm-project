// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-vertex -x metal -fmetal-entry=moon_vertex -emit-obj -Wno-c++11-narrowing -o %t.vs.spv %S/Inputs/moon.metal
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-pixel -x metal -fmetal-entry=moon_fragment -emit-obj -Wno-c++11-narrowing -o %t.fs.spv %S/Inputs/moon.metal
//
// The full Mooncraft height-field ray-marcher (kMoonMSL) — the P1
// resource-threading corpus: textures, the sampler, and the constant
// Uniforms reference are passed as arguments through a deep helper chain
// (terrain/shade/gradLevel/terrainGrad...), explicit-LOD samples with
// level(lodAt(...)), get_width() texel-size math, a direct [[position]]
// fragment input, dynamic u.bases[bi] indexing, and heavy multi-break
// loops with early-return debug modes.
//
// Both entries must compile from the FULL file. This is also the
// compile-time regression for the exponential convergence-region path
// walk (67+ minutes before memoization; well under a second after) and
// for the lazy emission of unreferenced helpers (the vertex module must
// not code-generate the fragment's helper bodies).
