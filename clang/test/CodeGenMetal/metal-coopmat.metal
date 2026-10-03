// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-compute -x metal -fmetal-entry=gemm_tile -emit-llvm -o - %s | FileCheck %s
//
// WinCatalyst cooperative matrices (Core AI slice 5 part 2): wc_coop_matrix<T,
// R, C, Use> is an empty marker record in the AST and the SPIR-V cooperative-
// matrix target type in IR (scope Subgroup = 3); the wc_coop_* calls become the
// __spirv_CooperativeMatrix*KHR wrapper builtins (load: ptr, layout, stride;
// store: ptr, matrix, layout, stride; mul_add: a, b, c; fill: CompositeConstruct),
// and a coop value copies as a load + store of the target type -- never the
// empty-class no-op. One 16x16x16 tile, f16 inputs, f32 accumulation (the shape
// the RTX 3050 reports).
#include <metal_stdlib>
using namespace metal;

struct GemmParams {
    uint K;
    uint lda, ldb, ldc;
};

typedef wc_coop_matrix<half, 16, 16, 0> TileA;
typedef wc_coop_matrix<half, 16, 16, 1> TileB;
typedef wc_coop_matrix<float, 16, 16, 2> TileC;

// CHECK: define void @gemm_tile()
// CHECK: call {{.*}}target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) @_Z26__spirv_CompositeConstruct_wc{{[A-Za-z0-9_]*}}(float 0.000000e+00)
// The accumulator is an SSA value by now (the Metal pipeline runs SROA at the
// start of O0, so the copy hook's load/store pairs fold into a phi):
// CHECK: phi target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2)
// CHECK: call {{.*}}target("spirv.CooperativeMatrixKHR", half, 3, 16, 16, 0) @_Z32__spirv_CooperativeMatrixLoadKHR_wc{{[A-Za-z0-9_]*}}(ptr addrspace(11) {{.*}}, i32 0, i32 {{.*}})
// CHECK: call {{.*}}target("spirv.CooperativeMatrixKHR", half, 3, 16, 16, 1) @_Z32__spirv_CooperativeMatrixLoadKHR_wc{{[A-Za-z0-9_]*}}(ptr addrspace(11) {{.*}}, i32 0, i32 {{.*}})
// CHECK: call {{.*}}target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) @_Z34__spirv_CooperativeMatrixMulAddKHR_wc{{[A-Za-z0-9_]*}}(target("spirv.CooperativeMatrixKHR", half, 3, 16, 16, 0) {{.*}}, target("spirv.CooperativeMatrixKHR", half, 3, 16, 16, 1) {{.*}}, target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) {{.*}})
// CHECK: call {{.*}}void @_Z33__spirv_CooperativeMatrixStoreKHR_wc{{[A-Za-z0-9_]*}}(ptr addrspace(11) {{.*}}, target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) {{.*}}, i32 0, i32 {{.*}})
// CHECK-NOT: llvm.memcpy
kernel void gemm_tile(constant GemmParams& p [[buffer(0)]],
                      const device half* A [[buffer(1)]],
                      const device half* B [[buffer(2)]],
                      device float* C [[buffer(3)]],
                      uint2 tile [[threadgroup_position_in_grid]]) {
    TileC acc = wc_coop_fill<float, 16, 16, 2>(0.0f);
    const uint row0 = tile.y * 16, col0 = tile.x * 16;
    for (uint k = 0; k < p.K; k += 16) {
        TileA a = wc_coop_load<half, 16, 16, 0>(&A[row0 * p.lda + k], p.lda, wc_coop_row_major);
        TileB b = wc_coop_load<half, 16, 16, 1>(&B[k * p.ldb + col0], p.ldb, wc_coop_row_major);
        acc = wc_coop_mul_add(a, b, acc);
    }
    wc_coop_store(&C[row0 * p.ldc + col0], acc, p.ldc, wc_coop_row_major);
}
