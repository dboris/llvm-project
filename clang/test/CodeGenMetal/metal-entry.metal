// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-vertex -x metal -fmetal-entry=tri_vertex -emit-llvm -o - %s | FileCheck %s --check-prefix=VS
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-pixel -x metal -fmetal-entry=tri_fragment -emit-llvm -o - %s | FileCheck %s --check-prefix=FS
//
// WinCatalyst Metal SPIR-V ABI (Harmony Phase 3): entry points become void()
// wrappers over the (internalized, always-inlined) user function, marshalling
// [[vertex_id]]/[[buffer(N)]]/[[stage_in]] and the return value onto the
// SPIR-V interface: buffers lower through llvm.spv.resource handles with
// typed member GEPs; varyings become Input/Output globals with Location
// assigned in declaration order, [[position]] mapping to BuiltIn Position
// (vertex output) / FragCoord (fragment input). -emit-llvm here shows the
// module after the O0 always-inliner, so the user function has already been
// folded into the wrapper.
#include <metal_stdlib>
using namespace metal;

struct TriVertex {
    float4 position;
    float4 color;
};

struct RasterizerData {
    float4 position [[position]];
    float3 color;
};

// VS-DAG: @wc.vertex_id = external hidden addrspace(7) externally_initialized constant i32, !spirv.Decorations [[VTXID_MD:![0-9]+]]
// VS-DAG: @wc.position = external hidden addrspace(8) externally_initialized global <4 x float>, !spirv.Decorations [[POS_MD:![0-9]+]]
// VS-DAG: @wc.out.color = external hidden addrspace(8) externally_initialized global <3 x float>, !spirv.Decorations [[LOC_MD:![0-9]+]]
//
// VS: define void @tri_vertex() [[VS_ATTRS:#[0-9]+]]
// VS: load i32, ptr addrspace(7) @wc.vertex_id
// VS: call target("spirv.VulkanBuffer", [0 x %struct.TriVertex], 12, 0) @llvm.spv.resource.handlefrombinding{{.*}}(i32 0, i32 0, i32 1, i32 0, i1 false, ptr @.str.wcbuf)
// VS: call ptr addrspace(11) @llvm.spv.resource.getpointer
// VS: getelementptr inbounds nuw %struct.TriVertex, ptr addrspace(11)
// VS: store <4 x float> {{.*}}, ptr addrspace(8) @wc.position
// VS: store <3 x float> {{.*}}, ptr addrspace(8) @wc.out.color
// VS: attributes [[VS_ATTRS]] = {{.*}}"hlsl.shader"="vertex"
// VS-DAG: [[VTXID_MD]] = !{[[VTXID_OPS:![0-9]+]]}
// VS-DAG: [[VTXID_OPS]] = !{i32 11, i32 42}
vertex RasterizerData tri_vertex(uint vertexID [[vertex_id]],
                                 constant TriVertex* vertices [[buffer(0)]]) {
    RasterizerData out;
    out.position = float4(vertices[vertexID].position.xy, 0.0, 1.0);
    out.color = vertices[vertexID].color.rgb;
    return out;
}

// The fragment stage_in loads FragCoord for the [[position]] member and
// Location 0 for the color varying; the return value is the Location 0
// output. The vertex entry above must not be emitted in this module
// (-fmetal-entry selects one entry per module).
// FS-NOT: @tri_vertex
// FS-DAG: @wc.frag_coord = external hidden addrspace(7) externally_initialized constant <4 x float>, !spirv.Decorations [[FC_MD:![0-9]+]]
// FS-DAG: @wc.in.color = external hidden addrspace(7) externally_initialized constant <3 x float>, !spirv.Decorations [[INLOC_MD:![0-9]+]]
// FS-DAG: @wc.color = external hidden addrspace(8) externally_initialized global <4 x float>, !spirv.Decorations [[OUTLOC_MD:![0-9]+]]
// FS: define void @tri_fragment() [[FS_ATTRS:#[0-9]+]]
// FS: load <4 x float>, ptr addrspace(7) @wc.frag_coord
// FS: load <3 x float>, ptr addrspace(7) @wc.in.color
// FS: store <4 x float> {{.*}}, ptr addrspace(8) @wc.color
// FS: attributes [[FS_ATTRS]] = {{.*}}"hlsl.shader"="pixel"
fragment float4 tri_fragment(RasterizerData in [[stage_in]]) {
    return float4(in.color, 1.0);
}
