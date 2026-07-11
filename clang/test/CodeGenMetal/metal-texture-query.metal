// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-pixel -x metal -fmetal-entry=q_frag -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
//
// get_width()/get_height() both lower to one OpImageQuerySizeLod at level 0
// (the spv.resource.imagequerysizelod intrinsic on a <2 x i32>), extracting
// component 0 for width and component 1 for height.
//
// Also the poison-laned vector-constant regression: a mixed vector literal
// whose leading lane is a constant and a later lane is dynamic —
// float2(0.0, dyn) — must seed from zeroinitializer, NOT poison, or the base
// const-folds to <float 0.0, poison> and the SPIR-V backend materializes it
// as a Kernel-only OpSpecConstantOp Bitcast (fails spirv-val for a shader).

#include <metal_stdlib>
using namespace metal;

fragment float4 q_frag(float4 pos [[position]],
                       texture2d<float> tex [[texture(0)]],
                       sampler s [[sampler(0)]]) {
  // CHECK: call {{.*}}@llvm.spv.resource.imagequerysizelod
  // CHECK: extractelement <2 x i32> {{.*}}, i32 0
  float w = float(tex.get_width());
  // CHECK: call {{.*}}@llvm.spv.resource.imagequerysizelod
  // CHECK: extractelement <2 x i32> {{.*}}, i32 1
  float h = float(tex.get_height());

  // The leading-constant / trailing-dynamic mixed init. Seeded from
  // zeroinitializer under Metal, so no poison lane can survive const-folding.
  // CHECK: insertelement <2 x float> zeroinitializer, float
  float2 v = float2(0.0, w + h);

  return tex.sample(s, v);
}
