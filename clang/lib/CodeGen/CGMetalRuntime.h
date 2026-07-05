//===----- CGMetalRuntime.h - Interface to Metal Runtimes -----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This provides an abstract class for Metal Shading Language code generation.
// The WinCatalyst Metal-on-Vulkan target (Harmony Phase 3): entry points are
// emitted as void() wrappers over the user function, marshalling Metal
// argument attributes ([[vertex_id]], [[buffer(N)]], [[stage_in]]) and the
// return value onto the SPIR-V interface ABI documented in WinCatalyst's
// <Metal/MTLLibrary.h>:
//   * descriptor set 0 = vertex stage, set 1 = fragment stage
//   * [[buffer(N)]]  -> SSBO binding N (llvm.spv.resource.handlefrombinding)
//   * varyings       -> Location 0,1,... in declaration order of the struct,
//                       [[position]] -> BuiltIn Position / FragCoord
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CGMETALRUNTIME_H
#define LLVM_CLANG_LIB_CODEGEN_CGMETALRUNTIME_H

#include "clang/AST/Decl.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/IRBuilder.h"

namespace llvm {
class Function;
class Value;
class GlobalVariable;
} // namespace llvm

namespace clang {
class FunctionDecl;
class ParmVarDecl;
class ArraySubscriptExpr;
class CXXMemberCallExpr;

namespace CodeGen {
class CodeGenModule;
class CodeGenFunction;
class LValue;
class RValue;

class CGMetalRuntime {
public:
  CGMetalRuntime(CodeGenModule &CGM) : CGM(CGM) {}

  /// Emit the void() entry wrapper for a vertex/fragment-qualified function.
  void emitEntryFunction(const FunctionDecl *FD, llvm::Function *Fn);

  /// True if PD is a [[buffer(N)]] parameter of a Metal entry function.
  static bool isBufferParam(const ParmVarDecl *PD);

  /// True if PD carries any Metal resource binding attribute
  /// ([[buffer(N)]], [[texture(N)]], [[sampler(N)]]). Resource parameters
  /// have no storage: every use is lowered through the SPIR-V resource
  /// intrinsics.
  static bool isResourceParam(const ParmVarDecl *PD);

  /// RValue for `tex.sample(smp, coord)` where tex is a [[texture(N)]]
  /// parameter — lowers to two handlefrombinding calls (image at 16+N,
  /// sampler at 24+N in the stage's set) + llvm.spv.resource.sampleimplicit.
  RValue emitTextureSampleCall(CodeGenFunction &CGF,
                               const CXXMemberCallExpr *E,
                               const ParmVarDecl *TexPD);

  /// LValue for `BufParam[Idx]` — lowers to
  /// llvm.spv.resource.handlefrombinding + llvm.spv.resource.getpointer.
  /// (A plain GEP would be pointer arithmetic, which logical SPIR-V cannot
  /// express — the backend silently drops the index; verified.)
  LValue emitBufferSubscriptLValue(CodeGenFunction &CGF,
                                   const ArraySubscriptExpr *E,
                                   const ParmVarDecl *PD);

  /// LValue for a direct reference to a buffer-attributed reference
  /// parameter (`constant T& u [[buffer(N)]]`) — element 0 of the SSBO.
  /// For non-reference (pointer) buffer params this diagnoses: raw pointer
  /// uses other than direct subscripts are not supported yet.
  LValue emitBufferParamDeclRefLValue(CodeGenFunction &CGF,
                                      const ParmVarDecl *PD);

private:
  CodeGenModule &CGM;

  /// Per-buffer-param name string globals (operand of handlefrombinding).
  llvm::DenseMap<const ParmVarDecl *, llvm::Constant *> BufferNameStrs;

  /// Emit handle + getpointer for element Idx of PD's SSBO. ElemTy is the
  /// Metal element type (pointee of the parameter).
  llvm::Value *emitBufferElementPtr(CodeGenFunction &CGF,
                                    const ParmVarDecl *PD, llvm::Value *Idx,
                                    QualType ElemTy);

  /// The descriptor set for the stage of the given entry function
  /// (0 = vertex, 1 = fragment).
  static unsigned stageDescriptorSet(const FunctionDecl *FD);
};

} // namespace CodeGen
} // namespace clang

#endif // LLVM_CLANG_LIB_CODEGEN_CGMETALRUNTIME_H
