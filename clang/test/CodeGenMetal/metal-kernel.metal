// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-compute -x metal -fmetal-entry=saxpy -emit-llvm -o - %s | FileCheck %s --check-prefix=CS
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-compute -x metal -fmetal-entry=reduce_sum -emit-llvm -o - %s | FileCheck %s --check-prefix=RS
//
// WinCatalyst Metal SPIR-V ABI, the compute third (Core AI slice 5 part 1): a
// `kernel` entry becomes a void() GLCompute wrapper ("hlsl.shader"="compute")
// with the PLACEHOLDER LocalSize 1,1,1 the Metal runtime rewrites per dispatch;
// [[buffer(N)]] lands in descriptor set 0 (the compute set) and a `device`
// pointer is a WRITABLE VulkanBuffer (Writable=1; `constant` stays 0); the
// thread-position parameters load from the compute BuiltIns through the
// per-component SPIR-V intrinsics; `threadgroup` locals become Workgroup
// (addrspace 3) statics; threadgroup_barrier is an OpControlBarrier wrapper
// call with AcquireRelease | WorkgroupMemory.
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint count;
    float alpha;
};

// CS: define void @saxpy() [[CS_ATTRS:#[0-9]+]]
// CS-DAG: call i32 @llvm.spv.thread.id.i32(i32 0)
// CS-DAG: call target("spirv.VulkanBuffer", [0 x %struct.Params], 12, 0) @llvm.spv.resource.handlefrombinding{{.*}}(i32 0, i32 0, i32 1, i32 0, i1 false, ptr @.str.wcres)
// CS-DAG: call target("spirv.VulkanBuffer", [0 x float], 12, 0) @llvm.spv.resource.handlefrombinding{{.*}}(i32 0, i32 1, i32 1, i32 0, i1 false, ptr @.str.wcres{{.*}})
// CS-DAG: call target("spirv.VulkanBuffer", [0 x float], 12, 1) @llvm.spv.resource.handlefrombinding{{.*}}(i32 0, i32 2, i32 1, i32 0, i1 false, ptr @.str.wcres{{.*}})
// CS: attributes [[CS_ATTRS]] = {{.*}}"hlsl.numthreads"="1,1,1"{{.*}}"hlsl.shader"="compute"
kernel void saxpy(constant Params& p [[buffer(0)]],
                  const device float* x [[buffer(1)]],
                  device float* y [[buffer(2)]],
                  uint gid [[thread_position_in_grid]]) {
    if (gid >= p.count) return;
    y[gid] = p.alpha * x[gid] + y[gid];
}

// A threadgroup reduction: the local array is a Workgroup-storage static, the
// barrier the ControlBarrier wrapper (Workgroup scope 2 twice, semantics
// AcquireRelease 0x8 | WorkgroupMemory 0x100 = 264), and all three position
// parameters appear (grid position as a vector, the flat index as a scalar).
// RS: @{{.*}}partial{{.*}} = internal addrspace(3) global [64 x float] undef
// RS: define void @reduce_sum() [[RS_ATTRS:#[0-9]+]]
// RS-DAG: call i32 @llvm.spv.thread.id.i32(i32 0)
// RS-DAG: call i32 @llvm.spv.thread.id.i32(i32 1)
// RS-DAG: call i32 @llvm.spv.flattened.thread.id.in.group()
// RS-DAG: call i32 @llvm.spv.group.id.i32(i32 0)
// RS: call {{.*}}void @_Z22__spirv_ControlBarrieriii(i32 2, i32 2, i32 264)
// RS: attributes [[RS_ATTRS]] = {{.*}}"hlsl.shader"="compute"
kernel void reduce_sum(const device float* x [[buffer(0)]],
                       device float* out [[buffer(1)]],
                       uint2 gid [[thread_position_in_grid]],
                       uint lid [[thread_index_in_threadgroup]],
                       uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float partial[64];
    partial[lid] = x[gid.x];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        float s = 0.0f;
        for (uint i = 0; i < 64; i++) s += partial[i];
        out[group.x] = s;
    }
}
