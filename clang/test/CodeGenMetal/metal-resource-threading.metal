// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-pixel -x metal -fmetal-entry=rt_fragment -emit-llvm -Wno-c++11-narrowing -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-pixel -x metal -fmetal-entry=rt_fragment -emit-obj -Wno-c++11-narrowing -o %t.spv %s
//
// Resources thread through helper calls AS VALUES (P1 resource threading):
// metal::texture2d<T>/metal::sampler lower to the SPIR-V handle
// target-ext-types, `constant T&` to the element-0 StorageBuffer pointer.
// The entry wrapper materializes each handle from its binding ONCE
// (buffer N, texture 16+N, sampler 24+N in the stage's set) and passes it
// as the call argument; every user function is internal + always_inline,
// so -emit-llvm shows the module after the O0 always-inliner with the
// handles feeding the resource intrinsics directly. A direct [[position]]
// parameter is the FragCoord builtin. get_width() is an
// OpImageQuerySizeLod at level 0 (component 0), sample(s, uv, level(l))
// the explicit-LOD image sample. u.items[i] below the threaded element
// pointer must stay a TYPED, zero-rooted GEP (a legal dynamic
// OpAccessChain), and each texture access is evaluated exactly once —
// vector arguments of a float4(...) construction are NOT re-evaluated
// per lane.
#include <metal_stdlib>
using namespace metal;

struct Uniforms {
  float2 res;
  float scale;
  float pad;
  int count;
  int p0, p1, p2;
  float4 items[8];
};

float pickScale(constant Uniforms& u, int i) {
  return u.items[i].x * u.scale;
}

float3 fetch(texture2d<float> t, sampler s, float2 uv, constant Uniforms& u) {
  float w = float(t.get_width());
  float3 a = t.sample(s, uv).rgb;
  float3 b = t.sample(s, uv, level(w / 256.0f)).rgb;
  return mix(a, b, 0.5f) * pickScale(u, u.count);
}

// CHECK: @wc.frag_coord = external hidden addrspace(7) externally_initialized constant <4 x float>
// CHECK: define void @rt_fragment()
// CHECK: load <4 x float>, ptr addrspace(7) @wc.frag_coord
//
// One handle per resource, materialized in the wrapper from its binding.
// CHECK: call target("spirv.VulkanBuffer", [0 x %struct.Uniforms], 12, 0) @llvm.spv.resource.handlefrombinding{{.*}}(i32 1, i32 0,
// CHECK: [[UPTR:%[0-9]+]] = call ptr addrspace(11) @llvm.spv.resource.getpointer{{.*}}(target("spirv.VulkanBuffer", [0 x %struct.Uniforms], 12, 0) {{%[0-9]+}}, i32 0)
// CHECK: [[IMG:%[0-9]+]] = call target("spirv.Image", float, 1, 2, 0, 0, 1, 0) @llvm.spv.resource.handlefrombinding{{.*}}(i32 1, i32 16,
// CHECK: [[SMP:%[0-9]+]] = call target("spirv.Sampler") @llvm.spv.resource.handlefrombinding{{.*}}(i32 1, i32 24,
//
// get_width: query size of mip 0, component 0.
// CHECK: [[SZ:%[0-9]+]] = call <2 x i32> @llvm.spv.resource.imagequerysizelod{{.*}}(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) [[IMG]], i32 0)
// CHECK: extractelement <2 x i32> [[SZ]], i32 0
//
// The samples consume the THREADED handle values — exactly one implicit
// and one explicit-LOD sample (no per-lane re-evaluation).
// CHECK: call <4 x float> @llvm.spv.resource.sampleimplicit{{.*}}(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) [[IMG]], target("spirv.Sampler") [[SMP]],
// CHECK-NOT: call <4 x float> @llvm.spv.resource.sampleimplicit
// CHECK: call <4 x float> @llvm.spv.resource.sampleexplicitlod{{.*}}(target("spirv.Image", float, 1, 2, 0, 0, 1, 0) [[IMG]], target("spirv.Sampler") [[SMP]], <2 x float> {{.*}}, float
// CHECK-NOT: call <4 x float> @llvm.spv.resource.sampleexplicitlod
//
// Dynamic trailing index below the threaded element pointer: typed GEPs.
// CHECK: getelementptr inbounds nuw %struct.Uniforms, ptr addrspace(11)
// CHECK: getelementptr inbounds [8 x <4 x float>], ptr addrspace(11) {{.*}}, i64 0, i64
fragment float4 rt_fragment(float4 fragCoord [[position]],
                            constant Uniforms& u [[buffer(0)]],
                            texture2d<float> tex [[texture(0)]],
                            sampler smp [[sampler(0)]]) {
  return float4(fetch(tex, smp, fragCoord.xy / u.res, u), 1.0);
}
