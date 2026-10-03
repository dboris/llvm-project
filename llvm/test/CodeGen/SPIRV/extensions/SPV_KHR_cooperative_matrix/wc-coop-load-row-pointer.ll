; RUN: llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute --spirv-ext=+SPV_KHR_cooperative_matrix %s -o - | FileCheck %s
;
; WinCatalyst (Core AI slice 5 part 3). MSL `wc_coop_load(&tile[r][0], 5, ...)`
; on `threadgroup uint4 tile[32][5]`: the all-zero trailing GEP index is folded
; away, so the pointer reaches the builtin typed as a pointer to the whole ROW
; ([5 x <4 x i32>]). With the Shader capability a cooperative-matrix pointer
; must point to a scalar or vector -- the unit of the stride -- so the backend
; descends through element 0: the load must use a pointer to <4 x i32>, not the
; row (which the NVIDIA driver accepted, measuring the stride in rows).
;
; The same fold on a plain store (`tile[lid.x][0] = v`) was UNREACHABLE in
; SPIRVLegalizePointerCast ("Unsupported ptrcast use in store"): llc compiling
; this file at all is that half of the test.
;
; Generated from the MSL with the fork clang (-x metal -fmetal-entry=row0 -emit-llvm).

; CHECK-DAG: %[[#I32:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#V4:]] = OpTypeVector %[[#I32]] 4
; CHECK-DAG: %[[#ROWTY:]] = OpTypeArray %[[#V4]] %[[#]]
; CHECK-DAG: %[[#PROW:]] = OpTypePointer Workgroup %[[#ROWTY]]
; CHECK-DAG: %[[#PV4:]] = OpTypePointer Workgroup %[[#V4]]
; CHECK-DAG: %[[#ZERO:]] = OpConstant %[[#I32]] 0
; CHECK: %[[#SROW:]] = OpInBoundsAccessChain %[[#PROW]] %[[#]] %[[#]]
; CHECK: %[[#SELEM:]] = OpInBoundsAccessChain %[[#PV4]] %[[#SROW]] %[[#ZERO]]
; CHECK: OpStore %[[#SELEM]] %[[#]]
; CHECK: %[[#ROW:]] = OpInBoundsAccessChain %[[#PROW]] %[[#]] %[[#]]
; CHECK: %[[#ELEM:]] = OpInBoundsAccessChain %[[#PV4]] %[[#ROW]] %[[#ZERO]]
; CHECK: OpCooperativeMatrixLoadKHR %[[#]] %[[#ELEM]] %[[#]] %[[#]]
; A pointer that already points to a scalar is passed through unchanged:
; CHECK: OpCooperativeMatrixStoreKHR %[[#]] %[[#]] %[[#]] %[[#]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G10"
target triple = "spirv-unknown-vulkan1.3-compute"

%struct.P = type { i32 }

@.str.wcres = private unnamed_addr constant [2 x i8] c"p\00", align 1
@_ZZ4row0RU4AS111PPU4AS11fDv2_jE4tile = internal addrspace(3) global [32 x [5 x <4 x i32>]] undef, align 16
@.str.wcres.1 = private unnamed_addr constant [2 x i8] c"C\00", align 1

; Function Attrs: convergent noinline
define void @row0() #0 {
entry:
  %0 = call token @llvm.experimental.convergence.entry()
  %1 = call target("spirv.VulkanBuffer", [0 x %struct.P], 12, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0s_struct.Ps_12_0t(i32 0, i32 0, i32 1, i32 0, i1 false, ptr @.str.wcres)
  %2 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0s_struct.Ps_12_0t(target("spirv.VulkanBuffer", [0 x %struct.P], 12, 0) %1, i32 0)
  %3 = call i32 @llvm.spv.thread.id.in.group.i32(i32 0)
  %4 = insertelement <2 x i32> poison, i32 %3, i32 0
  %5 = call i32 @llvm.spv.thread.id.in.group.i32(i32 1)
  %6 = insertelement <2 x i32> %4, i32 %5, i32 1
  %splat.splatinsert.i = insertelement <4 x i32> poison, i32 %3, i64 0
  %splat.splat.i = shufflevector <4 x i32> %splat.splatinsert.i, <4 x i32> poison, <4 x i32> zeroinitializer
  %idxprom.i = zext i32 %3 to i64
  %arrayidx.i = getelementptr inbounds nuw [32 x [5 x <4 x i32>]], ptr addrspace(3) @_ZZ4row0RU4AS111PPU4AS11fDv2_jE4tile, i64 0, i64 %idxprom.i
  store <4 x i32> %splat.splat.i, ptr addrspace(3) %arrayidx.i, align 16
  call spir_func void @_Z22__spirv_ControlBarrieriii(i32 2, i32 2, i32 264) #5 [ "convergencectrl"(token %0) ]
  %7 = load i32, ptr addrspace(11) %2, align 4
  %idxprom2.i = zext i32 %7 to i64
  %arrayidx3.i = getelementptr inbounds nuw [32 x [5 x <4 x i32>]], ptr addrspace(3) @_ZZ4row0RU4AS111PPU4AS11fDv2_jE4tile, i64 0, i64 %idxprom2.i
  %wc.coop.load5.i = call spir_func target("spirv.CooperativeMatrixKHR", half, 3, 16, 16, 0) @_Z32__spirv_CooperativeMatrixLoadKHR_wcptr_addrspace_3_i32i32_rtarget__spirv_CooperativeMatrixKHR___half__3__16__16__0_(ptr addrspace(3) %arrayidx3.i, i32 0, i32 5) #5 [ "convergencectrl"(token %0) ]
  %wc.coop.fill6.i = call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) @_Z26__spirv_CompositeConstruct_wcfloat_rtarget__spirv_CooperativeMatrixKHR___float__3__16__16__2_(float 0.000000e+00) #5 [ "convergencectrl"(token %0) ]
  %8 = call target("spirv.VulkanBuffer", [0 x float], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0f32_12_1t(i32 0, i32 1, i32 1, i32 0, i1 false, ptr @.str.wcres.1)
  %9 = call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0f32_12_1t(target("spirv.VulkanBuffer", [0 x float], 12, 1) %8, i32 0)
  call spir_func void @_Z33__spirv_CooperativeMatrixStoreKHR_wcptr_addrspace_11_target__spirv_CooperativeMatrixKHR___float__3__16__16__2_i32i32(ptr addrspace(11) %9, target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %wc.coop.fill6.i, i32 0, i32 16) #5 [ "convergencectrl"(token %0) ]
  ret void
}

; Function Attrs: convergent nocallback nofree nosync nounwind willreturn memory(none)
declare token @llvm.experimental.convergence.entry() #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare target("spirv.VulkanBuffer", [0 x %struct.P], 12, 0) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0s_struct.Ps_12_0t(i32, i32, i32, i32, i1, ptr) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0s_struct.Ps_12_0t(target("spirv.VulkanBuffer", [0 x %struct.P], 12, 0), i32) #2

; Function Attrs: nounwind willreturn memory(none)
declare i32 @llvm.spv.thread.id.in.group.i32(i32) #3

; Function Attrs: convergent
declare void @_Z22__spirv_ControlBarrieriii(i32, i32, i32) #4

; Function Attrs: convergent
declare target("spirv.CooperativeMatrixKHR", half, 3, 16, 16, 0) @_Z32__spirv_CooperativeMatrixLoadKHR_wcptr_addrspace_3_i32i32_rtarget__spirv_CooperativeMatrixKHR___half__3__16__16__0_(ptr addrspace(3), i32, i32) #4

; Function Attrs: convergent
declare target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) @_Z26__spirv_CompositeConstruct_wcfloat_rtarget__spirv_CooperativeMatrixKHR___float__3__16__16__2_(float) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare target("spirv.VulkanBuffer", [0 x float], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0f32_12_1t(i32, i32, i32, i32, i1, ptr) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(none)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0f32_12_1t(target("spirv.VulkanBuffer", [0 x float], 12, 1), i32) #2

; Function Attrs: convergent
declare void @_Z33__spirv_CooperativeMatrixStoreKHR_wcptr_addrspace_11_target__spirv_CooperativeMatrixKHR___float__3__16__16__2_i32i32(ptr addrspace(11), target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2), i32, i32) #4

attributes #0 = { convergent noinline "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(none) }
attributes #3 = { nounwind willreturn memory(none) }
attributes #4 = { convergent }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang"}
