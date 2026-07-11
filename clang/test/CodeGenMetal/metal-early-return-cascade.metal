// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-pixel -x metal -fmetal-entry=red_fragment -S -Wno-c++11-narrowing -o - %s | FileCheck %s
//
// Structurizer regression: a cascade of early returns inside a condition
// used to produce invalid structured control flow — the construct
// membership walk (partialOrderVisit) stopped the whole visit at the
// merge's rank, truncating the block set; the exit funneling then rewired
// already-fixed inner constructs' branches (multi-level merge exits), and
// the single-exit dispatch emitted an OpSwitch that can never carry a
// valid OpSelectionMerge (its targets are merges of different construct
// levels).
//
// The exit dispatches must be chains of two-way conditional branches:
// CHECK-NOT: OpSwitch
#include <metal_stdlib>
using namespace metal;

struct U { float4 a; int debug; int p0, p1, p2; };

fragment float4 red_fragment(float4 fc [[position]],
                             constant U& u [[buffer(0)]]) {
  float3 col;
  if (fc.x > 0.0) {
    float d = fc.y;
    if (u.debug == 1) return float4(d, 0.0, 0.0, 1.0);
    if (u.debug == 2) return float4(d, d, 0.0, 1.0);
    if (u.debug == 3) return float4(d, 0.0, d, 1.0);
    col = float3(d);
  } else {
    col = float3(0.5);
  }
  return float4(col.x, col.y, col.z, 1.0);
}
