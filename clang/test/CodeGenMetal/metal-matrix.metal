// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-vertex -x metal -fmetal-entry=mvp_vertex -emit-llvm -o - %s | FileCheck %s
//
// float4x4 * float4 lowers to llvm.spv.matrix4.times.vector (the native
// OpMatrixTimesVector) with the four columns loaded from the constant
// buffer, so drivers use the same arithmetic path as for GLSL-built
// shaders. `constant`/`device` map to the StorageBuffer address space (11)
// on this target (SPIRVMetalMap).
#include <metal_stdlib>
using namespace metal;

struct Uniforms {
    float4x4 mvp;
};

struct VOut {
    float4 position [[position]];
};

// CHECK: define void @mvp_vertex()
// CHECK: call target("spirv.VulkanBuffer", [0 x %struct.Uniforms], 12, 0) @llvm.spv.resource.handlefrombinding{{.*}}(i32 0, i32 1,
// CHECK: call ptr addrspace(11) @llvm.spv.resource.getpointer
// CHECK-COUNT-4: load <4 x float>, ptr addrspace(11)
// CHECK: call <4 x float> @llvm.spv.matrix4.times.vector.v4f32(<4 x float> {{.*}}, <4 x float> {{.*}}, <4 x float> {{.*}}, <4 x float> {{.*}}, <4 x float>
vertex VOut mvp_vertex(uint vid [[vertex_id]],
                       constant float4* positions [[buffer(0)]],
                       constant Uniforms& uniforms [[buffer(1)]]) {
    VOut out;
    out.position = uniforms.mvp * positions[vid];
    return out;
}
