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
// argument attributes ([[vertex_id]], [[buffer(N)]], [[texture(N)]],
// [[sampler(N)]], [[position]], [[stage_in]]) and the return value onto the
// SPIR-V interface ABI documented in WinCatalyst's <Metal/MTLLibrary.h>:
//   * descriptor set 0 = vertex stage, set 1 = fragment stage
//   * [[buffer(N)]]  -> SSBO binding N, [[texture(N)]] -> binding 16+N,
//     [[sampler(N)]] -> binding 24+N
//   * varyings       -> Location 0,1,... in declaration order of the struct,
//                       [[position]] -> BuiltIn Position / FragCoord
//
// Resources thread through helper calls AS VALUES (P1 resource threading):
// metal::texture2d<T> / metal::sampler lower to the SPIR-V handle
// target-ext-types, `constant T&` buffer references to the element-0
// StorageBuffer pointer. The entry wrapper materializes each handle from its
// binding once and passes it as the call argument; every user function is
// internal + always_inline, so after inlining the handles feed the resource
// intrinsics directly and no function-typed resource use survives.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CGMETALRUNTIME_H
#define LLVM_CLANG_LIB_CODEGEN_CGMETALRUNTIME_H

#include "CGValue.h"
#include "clang/AST/Decl.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/IRBuilder.h"
#include <optional>

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
class CXXOperatorCallExpr;

namespace CodeGen {
class CodeGenModule;
class CodeGenFunction;

class CGMetalRuntime {
public:
  CGMetalRuntime(CodeGenModule &CGM) : CGM(CGM) {}

  /// Emit the void() entry wrapper for a vertex/fragment-qualified function.
  void emitEntryFunction(const FunctionDecl *FD, llvm::Function *Fn);

  /// True if PD is a [[buffer(N)]] parameter of a Metal entry function.
  static bool isBufferParam(const ParmVarDecl *PD);

  /// True if T is the metal::texture2d<T> marker record.
  static bool isTextureRecord(QualType T);

  /// True if T is the metal::sampler marker record.
  static bool isSamplerRecord(QualType T);

  /// True if T is a Metal resource record that lowers to a SPIR-V handle
  /// value (texture2d / sampler).
  static bool isResourceRecord(QualType T);

  /// The LLVM handle type for a Metal resource record: spirv.Image for
  /// metal::texture2d<T>, spirv.Sampler for metal::sampler; nullptr if T is
  /// not a resource record. Hooked into CodeGenTypes::ConvertType so both
  /// entry and helper parameters of these types carry the handle.
  llvm::Type *convertResourceRecordType(QualType T);

  /// RValue for a member call on a metal::texture2d parameter (entry OR
  /// helper — the handle is the parameter's value):
  ///   tex.sample(s, uv)             -> llvm.spv.resource.sampleimplicit
  ///   tex.sample(s, uv, level(l))   -> llvm.spv.resource.sampleexplicitlod
  ///   tex.get_width()               -> llvm.spv.resource.imagequerysizelod
  RValue emitTextureMemberCall(CodeGenFunction &CGF,
                               const CXXMemberCallExpr *E);

  /// The handle value for a texture/sampler expression used as a call
  /// argument: peels the by-value copy down to the parameter and loads its
  /// handle. Hooked into EmitCallArg so resource records are forwarded as
  /// SSA values, never as memory aggregates (a memcpy of a handle type is
  /// not expressible in logical SPIR-V).
  llvm::Value *emitResourceCallArg(CodeGenFunction &CGF, const Expr *E);

  /// If E is metal::operator*(float4x4, float4), lower it to
  /// llvm.spv.matrix4.times.vector (the real OpMatrixTimesVector) so drivers
  /// run their native matrix arithmetic — bit-identical with GLSL-built
  /// shaders. Returns std::nullopt when E is some other operator call.
  std::optional<RValue> tryEmitMatrixVectorMul(CodeGenFunction &CGF,
                                               const CXXOperatorCallExpr *E);

  /// LValue for `BufParam[Idx]` on a raw pointer buffer parameter — lowers to
  /// llvm.spv.resource.handlefrombinding + llvm.spv.resource.getpointer.
  /// (A plain GEP would be pointer arithmetic, which logical SPIR-V cannot
  /// express — the backend silently drops the index; verified.)
  LValue emitBufferSubscriptLValue(CodeGenFunction &CGF,
                                   const ArraySubscriptExpr *E,
                                   const ParmVarDecl *PD);

  /// Diagnose a direct use of a raw-pointer buffer parameter (anything but a
  /// subscript, which is intercepted). Reference buffer parameters do NOT
  /// come through here: they hold the element-0 pointer the entry wrapper
  /// passed and use ordinary codegen.
  LValue emitBufferParamDeclRefLValue(CodeGenFunction &CGF,
                                      const ParmVarDecl *PD);

private:
  CodeGenModule &CGM;

  /// Per-resource-param name string globals (operand of handlefrombinding).
  llvm::DenseMap<const ParmVarDecl *, llvm::Constant *> BufferNameStrs;

  /// Emit llvm.spv.resource.handlefrombinding for PD at (Set, Binding) with
  /// handle type HandleTy.
  llvm::Value *emitHandleFromBinding(llvm::IRBuilderBase &B,
                                     llvm::Type *HandleTy,
                                     const ParmVarDecl *PD, unsigned Set,
                                     unsigned Binding);

  /// Emit handle + getpointer for element Idx of PD's SSBO. ElemLLVMTy is
  /// the LLVM memory type of the Metal element type.
  llvm::Value *emitBufferElementPtr(llvm::IRBuilderBase &B,
                                    const FunctionDecl *FD,
                                    const ParmVarDecl *PD, llvm::Value *Idx,
                                    llvm::Type *ElemLLVMTy);

  /// The descriptor set for the stage of the given entry function
  /// (0 = vertex, 1 = fragment).
  static unsigned stageDescriptorSet(const FunctionDecl *FD);
};

} // namespace CodeGen
} // namespace clang

#endif // LLVM_CLANG_LIB_CODEGEN_CGMETALRUNTIME_H
