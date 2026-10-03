//===----- CGMetalRuntime.cpp - Interface to Metal Runtimes ---*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Metal Shading Language entry-point and resource codegen for the SPIR-V
// target (WinCatalyst Metal-on-Vulkan, Harmony Phase 3). Modeled on
// CGHLSLRuntime; see CGMetalRuntime.h for the interface ABI.
//
// IR-shape constraints this file honors (all verified against the SPIR-V
// backend, 21.1.6):
//  * every dynamic SSBO index goes through llvm.spv.resource.getpointer —
//    a GEP off a base element pointer is pointer arithmetic and the backend
//    silently drops the index;
//  * member access below the element pointer must stay TYPED, zero-rooted
//    GEPs (clang -O0 shape) — byte-offset GEPs hit an unimplemented case in
//    SPIRVLegalizePointerCast;
//  * vector constants must not contain poison lanes (lowered to an
//    OpSpecConstantOp Bitcast that requires the Kernel capability);
//  * extract/insertelement indices must be i32 (i64 drags in the Int64
//    capability, an optional Vulkan feature);
//  * texture/sampler handles must reach the sample/query intrinsics as SSA
//    values whose defs are the handlefrombinding intrinsics — the entry
//    wrapper materializes them once, they thread through helper calls as
//    ordinary arguments, and the always-inliner + pre-inline SROA collapse
//    the plumbing before the backend runs.
//
//===----------------------------------------------------------------------===//

#include "CGMetalRuntime.h"
#include "CGDebugInfo.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IntrinsicsSPIRV.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include <cctype>

using namespace clang;
using namespace CodeGen;
using namespace llvm;

static void addSPIRVDecoration(llvm::GlobalVariable *GV, unsigned Decoration,
                               unsigned Operand) {
  LLVMContext &Ctx = GV->getContext();
  IRBuilder<> B(Ctx);
  MDNode *Operands =
      MDNode::get(Ctx, {ConstantAsMetadata::get(B.getInt32(Decoration)),
                        ConstantAsMetadata::get(B.getInt32(Operand))});
  MDNode *DecorationList = MDNode::get(Ctx, {Operands});
  GV->addMetadata("spirv.Decorations", *DecorationList);
}

// Input (storage class 7) / Output (storage class 8) interface globals,
// decorated either BuiltIn (decoration 11) or Location (decoration 30).
static llvm::GlobalVariable *createInterfaceGlobal(llvm::Module &M,
                                                   llvm::Type *Ty,
                                                   const Twine &Name,
                                                   bool IsInput,
                                                   unsigned Decoration,
                                                   unsigned Operand) {
  auto *GV = new llvm::GlobalVariable(
      M, Ty, /*isConstant=*/IsInput, llvm::GlobalValue::ExternalLinkage,
      /*Initializer=*/nullptr, Name, /*insertBefore=*/nullptr,
      llvm::GlobalVariable::NotThreadLocal,
      /*AddressSpace=*/IsInput ? 7u : 8u, /*isExternallyInitialized=*/true);
  GV->setVisibility(llvm::GlobalValue::HiddenVisibility);
  addSPIRVDecoration(GV, Decoration, Operand);
  return GV;
}

static constexpr unsigned kBuiltInDecoration = 11;
static constexpr unsigned kLocationDecoration = 30;
static constexpr unsigned kBuiltInPosition = 0;
static constexpr unsigned kBuiltInVertexIndex = 42;
static constexpr unsigned kBuiltInFragCoord = 15;
// SPIR-V StorageClass::StorageBuffer, encoded in the VulkanBuffer type.
static constexpr unsigned kStorageBufferSC = 12;

bool CGMetalRuntime::isEntryFunction(const FunctionDecl *FD) {
  return FD && (FD->hasAttr<MetalVertexAttr>() ||
                FD->hasAttr<MetalFragmentAttr>() ||
                FD->hasAttr<MetalKernelAttr>());
}

unsigned CGMetalRuntime::stageDescriptorSet(const FunctionDecl *FD) {
  // <Metal/MTLLibrary.h>: descriptor set 0 = vertex stage, 1 = fragment; a
  // kernel's compute pipeline layout has the single set 0 (same offsets).
  return FD->hasAttr<MetalFragmentAttr>() ? 1u : 0u;
}

// A [[buffer(N)]] parameter's SSBO is writable when it names `device` memory
// that is not const-qualified (`device float* out`); `constant` buffers and
// `const device` views stay NonWritable (the backend decorates them so).
static bool isWritableBufferParam(const ParmVarDecl *PD) {
  QualType T = PD->getType();
  QualType Pointee;
  if (const auto *RefTy = T->getAs<clang::ReferenceType>())
    Pointee = RefTy->getPointeeType();
  else if (const auto *PtrTy = T->getAs<clang::PointerType>())
    Pointee = PtrTy->getPointeeType();
  else
    return false;
  return Pointee.getAddressSpace() == LangAS::opencl_global &&
         !Pointee.isConstQualified();
}

bool CGMetalRuntime::isBufferParam(const ParmVarDecl *PD) {
  return PD && PD->hasAttr<MetalBufferBindingAttr>();
}

// The marker records live directly in namespace metal (metal_stdlib).
static const RecordDecl *getMetalNamespaceRecord(QualType T) {
  const RecordDecl *RD = T->getAsRecordDecl();
  if (!RD)
    return nullptr;
  const auto *NS = dyn_cast<NamespaceDecl>(RD->getDeclContext());
  if (!NS || NS->getName() != "metal")
    return nullptr;
  return RD;
}

bool CGMetalRuntime::isTextureRecord(QualType T) {
  const RecordDecl *RD = getMetalNamespaceRecord(T);
  return RD && RD->getName() == "texture2d";
}

bool CGMetalRuntime::isSamplerRecord(QualType T) {
  const RecordDecl *RD = getMetalNamespaceRecord(T);
  return RD && RD->getName() == "sampler";
}

bool CGMetalRuntime::isResourceRecord(QualType T) {
  return isTextureRecord(T) || isSamplerRecord(T);
}

llvm::Type *CGMetalRuntime::convertResourceRecordType(QualType T) {
  // Not a RESOURCE (no binding, an ordinary value), but the same hook: the
  // cooperative-matrix marker record's IR type is the SPIR-V target type.
  if (llvm::Type *CM = convertCoopMatrixRecordType(T))
    return CM;
  if (isSamplerRecord(T))
    return llvm::TargetExtType::get(CGM.getLLVMContext(), "spirv.Sampler");
  if (isTextureRecord(T)) {
    // OpTypeImage: Dim2D(1), depth unspecified(2), non-arrayed,
    // single-sampled, sampled(1), format Unknown(0).
    const auto *Spec =
        cast<ClassTemplateSpecializationDecl>(T->getAsRecordDecl());
    QualType Elem = Spec->getTemplateArgs()[0].getAsType();
    llvm::Type *SampledTy = CGM.getTypes().ConvertType(Elem);
    return llvm::TargetExtType::get(CGM.getLLVMContext(), "spirv.Image",
                                    {SampledTy}, {1, 2, 0, 0, 1, 0});
  }
  return nullptr;
}

llvm::Value *CGMetalRuntime::emitHandleFromBinding(llvm::IRBuilderBase &B,
                                                   llvm::Type *HandleTy,
                                                   const ParmVarDecl *PD,
                                                   unsigned Set,
                                                   unsigned Binding) {
  llvm::Constant *&NameStr = BufferNameStrs[PD];
  if (!NameStr)
    NameStr = CGM.GetAddrOfConstantCString(PD->getNameAsString(), ".str.wcres")
                  .getPointer();
  llvm::Function *HandleFn = CGM.getIntrinsic(
      llvm::Intrinsic::spv_resource_handlefrombinding, {HandleTy});
  return B.CreateCall(HandleFn,
                      {B.getInt32(Set), B.getInt32(Binding), B.getInt32(1),
                       B.getInt32(0), B.getInt1(false), NameStr});
}

llvm::Value *CGMetalRuntime::emitBufferElementPtr(llvm::IRBuilderBase &B,
                                                  const FunctionDecl *FD,
                                                  const ParmVarDecl *PD,
                                                  llvm::Value *Idx,
                                                  llvm::Type *ElemLLVMTy) {
  auto *BindingAttr = PD->getAttr<MetalBufferBindingAttr>();
  unsigned Set = stageDescriptorSet(FD);
  unsigned Binding = BindingAttr->getIndex();

  auto *RuntimeArrTy = llvm::ArrayType::get(ElemLLVMTy, 0);
  auto *HandleTy = llvm::TargetExtType::get(
      CGM.getLLVMContext(), "spirv.VulkanBuffer", {RuntimeArrTy},
      {kStorageBufferSC, /*Writable=*/isWritableBufferParam(PD) ? 1u : 0u});

  llvm::Value *Handle = emitHandleFromBinding(B, HandleTy, PD, Set, Binding);

  llvm::Value *Idx32 = B.CreateZExtOrTrunc(Idx, B.getInt32Ty());
  auto *PtrTy = llvm::PointerType::get(CGM.getLLVMContext(), 11);
  llvm::Function *GetPtrFn = CGM.getIntrinsic(
      llvm::Intrinsic::spv_resource_getpointer, {PtrTy, HandleTy});
  return B.CreateCall(GetPtrFn, {Handle, Idx32});
}

LValue CGMetalRuntime::emitBufferSubscriptLValue(CodeGenFunction &CGF,
                                                 const ArraySubscriptExpr *E,
                                                 const ParmVarDecl *PD) {
  const auto *FD = dyn_cast_or_null<FunctionDecl>(CGF.CurFuncDecl);
  assert(FD && "buffer parameter outside a Metal entry function");
  llvm::Value *Idx = CGF.EmitScalarExpr(E->getIdx());
  QualType ElemTy = E->getType();
  llvm::Value *Ptr = emitBufferElementPtr(CGF.Builder, FD, PD, Idx,
                                          CGF.ConvertTypeForMem(ElemTy));
  CharUnits Align = CGF.getContext().getTypeAlignInChars(ElemTy);
  return CGF.MakeAddrLValue(
      Address(Ptr, CGF.ConvertTypeForMem(ElemTy), Align), ElemTy);
}

LValue CGMetalRuntime::emitBufferParamDeclRefLValue(CodeGenFunction &CGF,
                                                    const ParmVarDecl *PD) {
  // Only raw-pointer buffer params reach here (reference params carry the
  // element-0 pointer as an ordinary argument). Direct uses other than a
  // subscript (address taken, pointer arithmetic) are not representable.
  QualType T = PD->getType();
  CGM.Error(PD->getLocation(),
            "Metal pointer buffer parameters only support direct subscript "
            "access in this implementation");
  llvm::Type *Ty = CGF.ConvertType(T);
  CharUnits Align = CGF.getContext().getTypeAlignInChars(T);
  llvm::Value *Poison = llvm::PoisonValue::get(CGM.UnqualPtrTy);
  return CGF.MakeAddrLValue(Address(Poison, Ty, Align), T);
}

// Peel a by-value resource argument (CXXConstructExpr copy /
// MaterializeTemporaryExpr) down to the parameter it names.
static const ParmVarDecl *peelResourceParmRef(const Expr *E) {
  const Expr *Arg = E->IgnoreParenImpCasts();
  while (true) {
    if (const auto *CCE = dyn_cast<CXXConstructExpr>(Arg);
        CCE && CCE->getNumArgs() >= 1)
      Arg = CCE->getArg(0)->IgnoreParenImpCasts();
    else if (const auto *MTE = dyn_cast<MaterializeTemporaryExpr>(Arg))
      Arg = MTE->getSubExpr()->IgnoreParenImpCasts();
    else
      break;
  }
  if (const auto *DRE = dyn_cast<DeclRefExpr>(Arg))
    return dyn_cast<ParmVarDecl>(DRE->getDecl());
  return nullptr;
}

// Load the handle value of a texture/sampler parameter. The parameter's
// storage is an alloca of the handle type; pre-inline SROA promotes the
// load/store plumbing so the handlefrombinding def reaches the intrinsics.
static llvm::Value *loadResourceParamHandle(CodeGenFunction &CGF,
                                            const ParmVarDecl *PD,
                                            const Twine &Name) {
  return CGF.Builder.CreateLoad(CGF.GetAddrOfLocalVar(PD), Name);
}

llvm::Value *CGMetalRuntime::emitResourceCallArg(CodeGenFunction &CGF,
                                                 const Expr *E) {
  if (const ParmVarDecl *PD = peelResourceParmRef(E);
      PD && isResourceRecord(PD->getType()))
    return loadResourceParamHandle(CGF, PD, "wc.res.arg");
  CGM.Error(E->getExprLoc(),
            "Metal texture/sampler arguments must name a texture/sampler "
            "parameter in this implementation");
  return llvm::PoisonValue::get(CGF.ConvertType(E->getType()));
}

// Peel `level(l)` (functional cast / temporary / by-value copy) down to the
// float LOD expression and emit it.
static llvm::Value *emitLevelLodScalar(CodeGenFunction &CGF, const Expr *E) {
  const Expr *Arg = E->IgnoreParenImpCasts();
  while (true) {
    if (const auto *FCE = dyn_cast<CXXFunctionalCastExpr>(Arg)) {
      Arg = FCE->getSubExpr()->IgnoreParenImpCasts();
      continue;
    }
    if (const auto *MTE = dyn_cast<MaterializeTemporaryExpr>(Arg)) {
      Arg = MTE->getSubExpr()->IgnoreParenImpCasts();
      continue;
    }
    if (const auto *CCE = dyn_cast<CXXConstructExpr>(Arg);
        CCE && CCE->getNumArgs() >= 1) {
      const Expr *A0 = CCE->getArg(0);
      if (A0->getType()->isFloatingType())
        return CGF.EmitScalarExpr(A0);
      Arg = A0->IgnoreParenImpCasts();
      continue;
    }
    break;
  }
  return nullptr;
}

RValue CGMetalRuntime::emitTextureMemberCall(CodeGenFunction &CGF,
                                             const CXXMemberCallExpr *E) {
  llvm::Type *ResultTy = CGF.ConvertType(E->getType());
  const CXXMethodDecl *MD = E->getMethodDecl();
  StringRef Name = MD ? MD->getName() : StringRef();

  const ParmVarDecl *TexPD =
      peelResourceParmRef(E->getImplicitObjectArgument());
  if (!TexPD || !isTextureRecord(TexPD->getType())) {
    CGM.Error(E->getExprLoc(),
              "Metal texture member calls are only supported on texture "
              "parameters in this implementation");
    return RValue::get(llvm::PoisonValue::get(ResultTy));
  }
  llvm::Value *Img = loadResourceParamHandle(CGF, TexPD, "wc.tex");

  if ((Name == "get_width" || Name == "get_height") && E->getNumArgs() == 0) {
    // OpImageQuerySizeLod at level 0 -> (width, height); component 0 = width,
    // component 1 = height.
    auto *SizeTy = llvm::FixedVectorType::get(CGF.Builder.getInt32Ty(), 2);
    llvm::Function *QueryFn =
        CGM.getIntrinsic(llvm::Intrinsic::spv_resource_imagequerysizelod,
                         {SizeTy, Img->getType()});
    llvm::Value *Size =
        CGF.Builder.CreateCall(QueryFn, {Img, CGF.Builder.getInt32(0)});
    unsigned Comp = (Name == "get_width") ? 0 : 1;
    return RValue::get(CGF.Builder.CreateExtractElement(
        Size, CGF.Builder.getInt32(Comp),
        Comp == 0 ? "wc.tex.width" : "wc.tex.height"));
  }

  if (Name == "sample" && (E->getNumArgs() == 2 || E->getNumArgs() == 3)) {
    const ParmVarDecl *SamplerPD = peelResourceParmRef(E->getArg(0));
    if (!SamplerPD || !isSamplerRecord(SamplerPD->getType())) {
      CGM.Error(E->getExprLoc(),
                "the sampler argument of Metal texture sample() must name a "
                "sampler parameter in this implementation");
      return RValue::get(llvm::PoisonValue::get(ResultTy));
    }
    llvm::Value *Smp = loadResourceParamHandle(CGF, SamplerPD, "wc.smp");
    llvm::Value *Coord = CGF.EmitScalarExpr(E->getArg(1));

    if (E->getNumArgs() == 2) {
      llvm::Function *SampleFn = CGM.getIntrinsic(
          llvm::Intrinsic::spv_resource_sampleimplicit,
          {ResultTy, Img->getType(), Smp->getType(), Coord->getType()});
      return RValue::get(CGF.Builder.CreateCall(SampleFn, {Img, Smp, Coord}));
    }

    llvm::Value *Lod = emitLevelLodScalar(CGF, E->getArg(2));
    if (!Lod) {
      CGM.Error(E->getExprLoc(),
                "the LOD argument of Metal texture sample() must be "
                "level(<float>) in this implementation");
      return RValue::get(llvm::PoisonValue::get(ResultTy));
    }
    llvm::Function *SampleFn = CGM.getIntrinsic(
        llvm::Intrinsic::spv_resource_sampleexplicitlod,
        {ResultTy, Img->getType(), Smp->getType(), Coord->getType()});
    return RValue::get(
        CGF.Builder.CreateCall(SampleFn, {Img, Smp, Coord, Lod}));
  }

  CGM.Error(E->getExprLoc(), "unsupported Metal texture member call");
  return RValue::get(llvm::PoisonValue::get(ResultTy));
}

std::optional<RValue>
CGMetalRuntime::tryEmitMatrixVectorMul(CodeGenFunction &CGF,
                                       const CXXOperatorCallExpr *E) {
  const auto *FD = dyn_cast_or_null<FunctionDecl>(E->getCalleeDecl());
  if (!FD || FD->getOverloadedOperator() != OO_Star ||
      FD->getNumParams() != 2)
    return std::nullopt;
  const auto *NS = dyn_cast<NamespaceDecl>(FD->getDeclContext());
  if (!NS || NS->getName() != "metal")
    return std::nullopt;
  const auto *RefTy = FD->getParamDecl(0)->getType()->getAs<ReferenceType>();
  if (!RefTy)
    return std::nullopt;
  const RecordDecl *RD = RefTy->getPointeeType()->getAsRecordDecl();
  if (!RD || RD->getName() != "float4x4")
    return std::nullopt;

  LValue MatLV = CGF.EmitLValue(E->getArg(0));
  llvm::Type *MatTy = CGF.ConvertTypeForMem(RefTy->getPointeeType());
  llvm::Value *Cols[4];
  for (unsigned I = 0; I < 4; ++I) {
    Address ColAddr =
        CGF.Builder.CreateStructGEP(MatLV.getAddress(), I, "wc.mat.col");
    Cols[I] = CGF.Builder.CreateLoad(ColAddr);
  }
  (void)MatTy;
  llvm::Value *Vec = CGF.EmitScalarExpr(E->getArg(1));
  llvm::Function *MulFn = CGM.getIntrinsic(
      llvm::Intrinsic::spv_matrix4_times_vector, {Vec->getType()});
  return RValue::get(CGF.Builder.CreateCall(
      MulFn, {Cols[0], Cols[1], Cols[2], Cols[3], Vec}));
}

std::optional<RValue>
CGMetalRuntime::tryEmitThreadgroupBarrier(CodeGenFunction &CGF,
                                          const CallExpr *E) {
  const auto *FD = dyn_cast_or_null<FunctionDecl>(E->getCalleeDecl());
  // An operator / constructor callee has no simple identifier (getName()
  // asserts on it) -- the implicit copy-assignment of a coop-matrix value
  // arrives here as a CXXOperatorCallExpr.
  if (!FD || !FD->getIdentifier() || FD->getName() != "threadgroup_barrier" ||
      E->getNumArgs() != 1)
    return std::nullopt;
  const auto *NS = dyn_cast<NamespaceDecl>(FD->getDeclContext());
  if (!NS || NS->getName() != "metal")
    return std::nullopt;

  // The flags select which storage classes the barrier orders; the operand is
  // an OpControlBarrier literal, so it has to be a constant here.
  Expr::EvalResult Flags;
  if (!E->getArg(0)->EvaluateAsInt(Flags, CGM.getContext())) {
    CGM.Error(E->getExprLoc(),
              "the mem_flags argument of threadgroup_barrier must be a "
              "constant in this implementation");
    return RValue::get(nullptr);
  }
  const uint64_t F = Flags.Val.getInt().getZExtValue();
  // SPIR-V MemorySemantics: AcquireRelease 0x8, UniformMemory 0x40 (the
  // StorageBuffer class in a Vulkan shader = MSL `device`), WorkgroupMemory
  // 0x100 (`threadgroup`), ImageMemory 0x800 (`texture`). mem_none still
  // synchronizes execution. Scope: Workgroup = 2 for both operands.
  unsigned Sem = 0x8;
  if (F & 1)
    Sem |= 0x40;
  if (F & 2)
    Sem |= 0x100;
  if (F & 4)
    Sem |= 0x800;

  // `__spirv_ControlBarrier(int, int, int)`: the SPIR-V backend recognizes the
  // demangled name and emits OpControlBarrier with the operands as given
  // (SPIRVBuiltins "__spirv_ControlBarrier", a wrapper op); the HLSL intrinsic
  // (group_memory_barrier_with_group_sync) would drop the storage-class bits.
  llvm::LLVMContext &Ctx = CGM.getLLVMContext();
  llvm::Type *I32 = llvm::Type::getInt32Ty(Ctx);
  auto *FT = llvm::FunctionType::get(llvm::Type::getVoidTy(Ctx),
                                     {I32, I32, I32}, false);
  llvm::FunctionCallee Callee =
      CGM.getModule().getOrInsertFunction("_Z22__spirv_ControlBarrieriii", FT);
  if (auto *Fn = dyn_cast<llvm::Function>(Callee.getCallee()))
    Fn->setConvergent();
  llvm::Value *Scope = llvm::ConstantInt::get(I32, 2);
  CGF.EmitRuntimeCall(Callee, {Scope, Scope, llvm::ConstantInt::get(I32, Sem)});
  return RValue::get(nullptr);
}

// ---- cooperative matrices (WinCatalyst extension) ---------------------------
//
// metal::wc_coop_matrix<T, Rows, Cols, Use> is an empty marker record; its IR
// type is target("spirv.CooperativeMatrixKHR", T, Subgroup=3, Rows, Cols, Use)
// and its values live in allocas of that type (clang's ordinary locals), copied
// by emitCoopMatrixCopy. The four free functions are only declared in
// metal_stdlib; calls are rewritten here to the SPIR-V wrapper builtins the
// backend recognizes by demangled name (it reads the length-prefixed
// identifier and ignores the rest of the mangling -- hence the "_wc..."
// suffixes that keep one llvm::Function per operand-type combination):
//   wc_coop_load<T,R,C,U>(ptr, stride, layout)  -> __spirv_CooperativeMatrixLoadKHR(ptr, layout, stride)
//   wc_coop_store(ptr, m, stride, layout)        -> __spirv_CooperativeMatrixStoreKHR(ptr, m, layout, stride)
//   wc_coop_mul_add(a, b, c)                     -> __spirv_CooperativeMatrixMulAddKHR(a, b, c)
//   wc_coop_fill<T,R,C,U>(v)                     -> __spirv_CompositeConstruct(v)
// Strides are in elements; layout 0 = row-major, 1 = column-major.

static const ClassTemplateSpecializationDecl *getCoopMatrixSpec(QualType T) {
  const RecordDecl *RD = getMetalNamespaceRecord(T);
  if (!RD || RD->getName() != "wc_coop_matrix")
    return nullptr;
  return dyn_cast<ClassTemplateSpecializationDecl>(RD);
}

bool CGMetalRuntime::isCoopMatrixRecord(QualType T) {
  return getCoopMatrixSpec(T) != nullptr;
}

llvm::Type *CGMetalRuntime::convertCoopMatrixRecordType(QualType T) {
  const auto *Spec = getCoopMatrixSpec(T);
  if (!Spec)
    return nullptr;
  const TemplateArgumentList &Args = Spec->getTemplateArgs();
  if (Args.size() != 4)
    return nullptr;
  llvm::Type *ElemTy = CGM.getTypes().ConvertType(Args[0].getAsType());
  const unsigned Rows = Args[1].getAsIntegral().getZExtValue();
  const unsigned Cols = Args[2].getAsIntegral().getZExtValue();
  const unsigned Use = Args[3].getAsIntegral().getZExtValue();
  // SPIR-V Scope::Subgroup = 3 (the only scope VK_KHR_cooperative_matrix offers).
  return llvm::TargetExtType::get(CGM.getLLVMContext(),
                                  "spirv.CooperativeMatrixKHR", {ElemTy},
                                  {3u, Rows, Cols, Use});
}

void CGMetalRuntime::emitCoopMatrixCopy(CodeGenFunction &CGF, LValue Dest,
                                        LValue Src, QualType Ty) {
  llvm::Type *MatTy = CGF.ConvertTypeForMem(Ty);
  llvm::Value *V = CGF.Builder.CreateLoad(
      Src.getAddress().withElementType(MatTy), "wc.coop.copy");
  CGF.Builder.CreateStore(V, Dest.getAddress().withElementType(MatTy));
}

// Peel the by-value copy wrappers off a coop-matrix argument.
static const Expr *peelCoopArg(const Expr *E) {
  const Expr *Arg = E->IgnoreParenImpCasts();
  while (true) {
    if (const auto *CCE = dyn_cast<CXXConstructExpr>(Arg);
        CCE && CCE->getNumArgs() >= 1)
      Arg = CCE->getArg(0)->IgnoreParenImpCasts();
    else if (const auto *MTE = dyn_cast<MaterializeTemporaryExpr>(Arg))
      Arg = MTE->getSubExpr()->IgnoreParenImpCasts();
    else
      break;
  }
  return Arg;
}

std::optional<RValue>
CGMetalRuntime::tryEmitCoopMatrixCall(CodeGenFunction &CGF, const CallExpr *E,
                                      ReturnValueSlot ReturnValue) {
  const auto *FD = dyn_cast_or_null<FunctionDecl>(E->getCalleeDecl());
  if (!FD || !FD->getIdentifier())  // operators / constructors: not ours
    return std::nullopt;
  StringRef Name = FD->getName();
  if (!Name.starts_with("wc_coop_"))
    return std::nullopt;
  const auto *NS = dyn_cast<NamespaceDecl>(FD->getDeclContext());
  if (!NS || NS->getName() != "metal")
    return std::nullopt;

  llvm::LLVMContext &Ctx = CGM.getLLVMContext();
  llvm::Type *I32 = llvm::Type::getInt32Ty(Ctx);
  auto Fail = [&](const char *Msg) -> std::optional<RValue> {
    CGM.Error(E->getExprLoc(), Msg);
    return RValue::get(nullptr);
  };
  // A constant-expression i32 operand (the layout).
  auto ConstI32 = [&](const Expr *A) -> llvm::Value * {
    Expr::EvalResult R;
    if (!A->EvaluateAsInt(R, CGM.getContext()))
      return nullptr;
    return llvm::ConstantInt::get(I32, R.Val.getInt().getZExtValue());
  };
  // A coop-matrix operand as an SSA value: a glvalue is loaded in place, a
  // prvalue (a nested call) is materialized into a temporary first.
  auto CoopArg = [&](const Expr *A) -> llvm::Value * {
    const Expr *P = peelCoopArg(A);
    QualType T = P->getType();
    llvm::Type *MatTy = CGF.ConvertTypeForMem(T);
    if (P->isGLValue()) {
      LValue LV = CGF.EmitLValue(P);
      return CGF.Builder.CreateLoad(LV.getAddress().withElementType(MatTy),
                                    "wc.coop.arg");
    }
    AggValueSlot Slot = CGF.CreateAggTemp(T, "wc.coop.tmp");
    CGF.EmitAggExpr(P, Slot);
    return CGF.Builder.CreateLoad(Slot.getAddress().withElementType(MatTy),
                                  "wc.coop.arg");
  };
  // Hand a coop-matrix result back as an aggregate: into the caller's slot
  // when it gave one, else into a temporary.
  auto ReturnCoop = [&](llvm::Value *V, QualType RetTy) -> RValue {
    Address Dest = ReturnValue.isNull()
                       ? Address(CGF.CreateMemTemp(RetTy, "wc.coop.ret"))
                       : ReturnValue.getAddress();
    CGF.Builder.CreateStore(V, Dest.withElementType(V->getType()));
    return RValue::getAggregate(Dest);
  };
  // One llvm::Function per operand-type combination, named so the backend's
  // demangler still reads the builtin's identifier.
  auto Callee = [&](const char *Mangled, llvm::Type *Ret,
                    ArrayRef<llvm::Type *> Params) -> llvm::FunctionCallee {
    std::string N = Mangled;
    N += "_wc";
    for (llvm::Type *P : Params) {
      std::string S;
      llvm::raw_string_ostream OS(S);
      P->print(OS, /*IsForDebug=*/false, /*NoDetails=*/true);
      for (char Ch : S)
        N += (isalnum((unsigned char)Ch) ? Ch : '_');
    }
    if (Ret && !Ret->isVoidTy()) {
      std::string S;
      llvm::raw_string_ostream OS(S);
      Ret->print(OS, false, true);
      N += "_r";
      for (char Ch : S)
        N += (isalnum((unsigned char)Ch) ? Ch : '_');
    }
    auto *FT = llvm::FunctionType::get(Ret, Params, false);
    llvm::FunctionCallee FC = CGM.getModule().getOrInsertFunction(N, FT);
    if (auto *Fn = dyn_cast<llvm::Function>(FC.getCallee()))
      Fn->setConvergent();  // subgroup-collective operations
    return FC;
  };

  if (Name == "wc_coop_load") {
    if (E->getNumArgs() != 3)
      return Fail("wc_coop_load takes (pointer, stride, layout)");
    QualType RetTy = E->getType();
    llvm::Type *MatTy = CGF.ConvertTypeForMem(RetTy);
    if (!isa<llvm::TargetExtType>(MatTy))
      return Fail("wc_coop_load's result must be a wc_coop_matrix");
    llvm::Value *Ptr = CGF.EmitScalarExpr(E->getArg(0));
    llvm::Value *Stride = CGF.Builder.CreateZExtOrTrunc(
        CGF.EmitScalarExpr(E->getArg(1)), I32);
    llvm::Value *Layout = ConstI32(E->getArg(2));
    if (!Layout)
      return Fail("wc_coop_load's layout must be a constant");
    llvm::FunctionCallee F =
        Callee("_Z32__spirv_CooperativeMatrixLoadKHR", MatTy,
               {Ptr->getType(), I32, I32});
    llvm::Value *V = CGF.EmitRuntimeCall(F, {Ptr, Layout, Stride}, "wc.coop.load");
    return ReturnCoop(V, RetTy);
  }
  if (Name == "wc_coop_store") {
    if (E->getNumArgs() != 4)
      return Fail("wc_coop_store takes (pointer, matrix, stride, layout)");
    llvm::Value *Ptr = CGF.EmitScalarExpr(E->getArg(0));
    llvm::Value *Mat = CoopArg(E->getArg(1));
    llvm::Value *Stride = CGF.Builder.CreateZExtOrTrunc(
        CGF.EmitScalarExpr(E->getArg(2)), I32);
    llvm::Value *Layout = ConstI32(E->getArg(3));
    if (!Layout)
      return Fail("wc_coop_store's layout must be a constant");
    llvm::FunctionCallee F =
        Callee("_Z33__spirv_CooperativeMatrixStoreKHR",
               llvm::Type::getVoidTy(Ctx),
               {Ptr->getType(), Mat->getType(), I32, I32});
    CGF.EmitRuntimeCall(F, {Ptr, Mat, Layout, Stride});
    return RValue::get(nullptr);
  }
  if (Name == "wc_coop_mul_add") {
    if (E->getNumArgs() != 3)
      return Fail("wc_coop_mul_add takes (a, b, c)");
    llvm::Value *A = CoopArg(E->getArg(0));
    llvm::Value *B = CoopArg(E->getArg(1));
    llvm::Value *C = CoopArg(E->getArg(2));
    llvm::FunctionCallee F =
        Callee("_Z34__spirv_CooperativeMatrixMulAddKHR", C->getType(),
               {A->getType(), B->getType(), C->getType()});
    llvm::Value *V = CGF.EmitRuntimeCall(F, {A, B, C}, "wc.coop.mad");
    return ReturnCoop(V, E->getType());
  }
  if (Name == "wc_coop_fill") {
    if (E->getNumArgs() != 1)
      return Fail("wc_coop_fill takes (value)");
    QualType RetTy = E->getType();
    llvm::Type *MatTy = CGF.ConvertTypeForMem(RetTy);
    if (!isa<llvm::TargetExtType>(MatTy))
      return Fail("wc_coop_fill's result must be a wc_coop_matrix");
    llvm::Value *Val = CGF.EmitScalarExpr(E->getArg(0));
    llvm::FunctionCallee F =
        Callee("_Z26__spirv_CompositeConstruct", MatTy, {Val->getType()});
    llvm::Value *V = CGF.EmitRuntimeCall(F, {Val}, "wc.coop.fill");
    return ReturnCoop(V, RetTy);
  }
  return Fail("unknown wc_coop_* function");
}

// Load an input-interface value for one stage_in field / builtin parameter.
static llvm::Value *loadInterfaceInput(IRBuilder<> &B, llvm::Module &M,
                                       llvm::Type *Ty, const Twine &Name,
                                       unsigned Decoration, unsigned Operand) {
  auto *GV = createInterfaceGlobal(M, Ty, Name, /*IsInput=*/true, Decoration,
                                   Operand);
  return B.CreateLoad(Ty, GV);
}

void CGMetalRuntime::emitEntryFunction(const FunctionDecl *FD,
                                       llvm::Function *Fn) {
  llvm::Module &M = CGM.getModule();
  llvm::LLVMContext &Ctx = M.getContext();
  bool IsVertex = FD->hasAttr<MetalVertexAttr>();
  bool IsKernel = FD->hasAttr<MetalKernelAttr>();
  unsigned Set = stageDescriptorSet(FD);

  auto *EntryTy = llvm::FunctionType::get(llvm::Type::getVoidTy(Ctx), false);
  Function *EntryFn =
      Function::Create(EntryTy, Function::ExternalLinkage, FD->getName(), &M);

  // Function-level attributes carry over; the wrapper has no arguments.
  AttributeList NewAttrs = AttributeList::get(Ctx, AttributeList::FunctionIndex,
                                              Fn->getAttributes().getFnAttrs());
  EntryFn->setAttributes(NewAttrs);
  EntryFn->addFnAttr("hlsl.shader",
                     llvm::Triple::getEnvironmentTypeName(
                         IsKernel   ? llvm::Triple::Compute
                         : IsVertex ? llvm::Triple::Vertex
                                    : llvm::Triple::Pixel));
  if (IsKernel) {
    // Vulkan needs a LocalSize in the module; Metal chooses
    // threadsPerThreadgroup at DISPATCH time. Emit a 1,1,1 placeholder
    // (OpExecutionMode LocalSize 1 1 1) that the WinCatalyst Metal runtime
    // rewrites to the dispatch's size when it builds the VkPipeline for it
    // (<Metal/MTLLibrary.h>, the compute ABI). [[threads_per_threadgroup]] is
    // therefore NOT offered: the WorkgroupSize BuiltIn would read the
    // placeholder, and Vulkan requires it on a constant anyway.
    EntryFn->addFnAttr("hlsl.numthreads", "1,1,1");
  }
  // NEVER OptimizeNone, even at O0: the SPIR-V backend's structurizer
  // contract is RegToMem (demote cross-block values) -> structurize ->
  // PromoteMemoryToRegister, and the promote step skips optnone functions
  // while the demote step does not — leaving every demoted slot as a
  // Function-storage OpVariable, which for the threaded resource pointers
  // (ptr addrspace(11)) logical SPIR-V cannot express (VariablePointers).
  EntryFn->addFnAttr(llvm::Attribute::NoInline);

  // The user function must fold into the wrapper: logical SPIR-V has no
  // linkage, and the resource handle values passed below must reach the
  // resource intrinsics as SSA defs.
  Fn->setLinkage(GlobalValue::InternalLinkage);
  Fn->removeFnAttr(llvm::Attribute::OptimizeNone);
  Fn->removeFnAttr(llvm::Attribute::NoInline);
  Fn->addFnAttr(llvm::Attribute::AlwaysInline);

  BasicBlock *BB = BasicBlock::Create(Ctx, "entry", EntryFn);
  IRBuilder<> B(BB);
  llvm::SmallVector<Value *> Args;

  // The user function is emitted convergent; give the call site a matching
  // convergence anchor or the always-inliner refuses to fold it.
  SmallVector<llvm::OperandBundleDef, 1> OB;
  if (CGM.shouldEmitConvergenceTokens()) {
    // The user function will be marked convergent by the time it is defined;
    // the wrapper must match (the attr copy above ran before that happens).
    EntryFn->setConvergent();
    llvm::Value *Token = B.CreateIntrinsic(
        llvm::Intrinsic::experimental_convergence_entry, {});
    llvm::Value *BundleArgs[] = {Token};
    OB.emplace_back("convergencectrl", BundleArgs);
  }

  // Locations are assigned in declaration order of the varying struct,
  // skipping [[position]] members — identically for the vertex output and
  // the fragment stage_in, so the two stages pair up by construction.
  auto ForEachVaryingField = [&](QualType RecordTy,
                                 llvm::function_ref<void(const FieldDecl *,
                                                         unsigned /*Index*/,
                                                         int /*Location*/)>
                                     Callback) {
    const RecordDecl *RD = RecordTy->getAsRecordDecl();
    assert(RD && "varying type is not a struct");
    unsigned Index = 0;
    int NextLoc = 0;
    for (const FieldDecl *Field : RD->fields()) {
      int Loc = Field->hasAttr<MetalPositionAttr>() ? -1 : NextLoc++;
      Callback(Field, Index++, Loc);
    }
  };

  llvm::AllocaInst *SRetAlloca = nullptr;
  unsigned SRetOffset = 0;
  for (const auto &Param : Fn->args()) {
    if (Param.hasStructRetAttr()) {
      SRetOffset = 1;
      SRetAlloca = B.CreateAlloca(Param.getParamStructRetType(), nullptr,
                                  "wc.sret");
      Args.emplace_back(SRetAlloca);
      continue;
    }
    const ParmVarDecl *PD = FD->getParamDecl(Param.getArgNo() - SRetOffset);
    if (PD->hasAttr<MetalVertexIdAttr>()) {
      Args.push_back(loadInterfaceInput(B, M, Param.getType(), "wc.vertex_id",
                                        kBuiltInDecoration,
                                        kBuiltInVertexIndex));
      continue;
    }
    if (PD->hasAttr<MetalPositionAttr>()) {
      // float4 fragCoord [[position]] as a direct fragment input.
      Args.push_back(loadInterfaceInput(B, M, Param.getType(), "wc.frag_coord",
                                        kBuiltInDecoration,
                                        kBuiltInFragCoord));
      continue;
    }
    if (const auto *TexAttr = PD->getAttr<MetalTextureBindingAttr>()) {
      // <Metal/MTLLibrary.h>: within the stage's set, textures sit at 16+N.
      Args.push_back(emitHandleFromBinding(B, Param.getType(), PD, Set,
                                           16 + TexAttr->getIndex()));
      continue;
    }
    if (const auto *SmpAttr = PD->getAttr<MetalSamplerBindingAttr>()) {
      // ... and samplers at 24+N.
      Args.push_back(emitHandleFromBinding(B, Param.getType(), PD, Set,
                                           24 + SmpAttr->getIndex()));
      continue;
    }
    if (PD->hasAttr<MetalBufferBindingAttr>()) {
      if (const auto *RefTy = PD->getType()->getAs<ReferenceType>()) {
        // constant T& u [[buffer(N)]] — the element-0 pointer of the SSBO,
        // threaded as the reference argument. Member access (including a
        // dynamic trailing index) is then ordinary typed GEPs off it.
        llvm::Type *ElemTy =
            CGM.getTypes().ConvertTypeForMem(RefTy->getPointeeType());
        Args.push_back(
            emitBufferElementPtr(B, FD, PD, B.getInt32(0), ElemTy));
      } else {
        // Raw device pointers keep the no-storage model: never read, every
        // subscript is lowered per element through the resource intrinsics.
        Args.emplace_back(PoisonValue::get(Param.getType()));
      }
      continue;
    }
    if (PD->hasAttr<MetalStageInAttr>()) {
      QualType RecordTy = PD->getType();
      llvm::Type *RecTy = CGM.getTypes().ConvertTypeForMem(RecordTy);
      llvm::AllocaInst *Tmp = B.CreateAlloca(RecTy, nullptr, "wc.stage_in");
      ForEachVaryingField(RecordTy, [&](const FieldDecl *Field, unsigned Index,
                                        int Loc) {
        llvm::Type *FieldTy =
            CGM.getTypes().ConvertTypeForMem(Field->getType());
        llvm::Value *V =
            Loc < 0 ? loadInterfaceInput(B, M, FieldTy, "wc.frag_coord",
                                         kBuiltInDecoration, kBuiltInFragCoord)
                    : loadInterfaceInput(B, M, FieldTy,
                                         "wc.in." + Field->getName(),
                                         kLocationDecoration, (unsigned)Loc);
        B.CreateStore(V, B.CreateStructGEP(RecTy, Tmp, Index));
      });
      if (Param.getType()->isPointerTy()) {
        // Indirect aggregate. Strip byval: the callee is internal and
        // always-inlined, and the wrapper never reads the struct after the
        // call, so the pointer can be substituted directly — the byval copy
        // the inliner would insert becomes an OpCopyMemorySized, which
        // logical SPIR-V cannot express.
        if (Param.hasByValAttr())
          Fn->removeParamAttr(Param.getArgNo(), llvm::Attribute::ByVal);
        Args.push_back(Tmp);
      } else {
        Args.push_back(B.CreateLoad(RecTy, Tmp));
      }
      continue;
    }
    // Kernel thread-position parameters: `uint` / `uint2` / `uint3` loaded
    // from the compute BuiltIns component by component (the HLSL
    // SV_DispatchThreadID shape -- the intrinsics are per-component i32).
    {
      std::optional<llvm::Intrinsic::ID> VecID;
      if (PD->hasAttr<MetalThreadPositionInGridAttr>())
        VecID = llvm::Intrinsic::spv_thread_id;
      else if (PD->hasAttr<MetalThreadPositionInThreadgroupAttr>())
        VecID = llvm::Intrinsic::spv_thread_id_in_group;
      else if (PD->hasAttr<MetalThreadgroupPositionInGridAttr>())
        VecID = llvm::Intrinsic::spv_group_id;
      else if (PD->hasAttr<MetalThreadgroupsPerGridAttr>())
        VecID = llvm::Intrinsic::spv_num_workgroups;
      bool IsFlatIndex = PD->hasAttr<MetalThreadIndexInThreadgroupAttr>();
      if (VecID || IsFlatIndex) {
        llvm::Type *Ty = Param.getType();
        llvm::Type *ElemTy =
            isa<FixedVectorType>(Ty) ? cast<FixedVectorType>(Ty)->getElementType()
                                     : Ty;
        if (!IsKernel || !ElemTy->isIntegerTy(32) ||
            (IsFlatIndex && Ty != ElemTy)) {
          CGM.Error(PD->getLocation(),
                    "Metal thread-position parameters are kernel-only and "
                    "must be uint / uint2 / uint3 (uint for "
                    "[[thread_index_in_threadgroup]]) in this implementation");
          Args.emplace_back(PoisonValue::get(Ty));
          continue;
        }
        if (IsFlatIndex) {
          llvm::Function *F = CGM.getIntrinsic(
              llvm::Intrinsic::spv_flattened_thread_id_in_group);
          Args.push_back(B.CreateCall(F, {}, "wc.thread_index"));
          continue;
        }
        llvm::Function *F = CGM.getIntrinsic(*VecID, {B.getInt32Ty()});
        if (auto *VT = dyn_cast<FixedVectorType>(Ty)) {
          llvm::Value *V = PoisonValue::get(VT);
          for (unsigned I = 0; I < VT->getNumElements(); ++I)
            V = B.CreateInsertElement(V, B.CreateCall(F, {B.getInt32(I)}),
                                      B.getInt32(I));
          Args.push_back(V);
        } else {
          Args.push_back(B.CreateCall(F, {B.getInt32(0)}, "wc.thread_pos"));
        }
        continue;
      }
    }
    CGM.Error(PD->getLocation(),
              "unsupported Metal entry parameter: expected [[vertex_id]], "
              "[[position]], [[buffer(N)]], [[texture(N)]], [[sampler(N)]], "
              "[[stage_in]] or a kernel thread-position attribute");
    Args.emplace_back(PoisonValue::get(Param.getType()));
  }

  CallInst *CI = B.CreateCall(FunctionCallee(Fn), Args, OB);
  CI->setCallingConv(Fn->getCallingConv());

  // Marshal the return value onto the output interface.
  QualType RetTy = FD->getReturnType();
  auto StoreOutput = [&](llvm::Value *V, const Twine &Name, unsigned Decoration,
                         unsigned Operand) {
    auto *GV = createInterfaceGlobal(M, V->getType(), Name, /*IsInput=*/false,
                                     Decoration, Operand);
    B.CreateStore(V, GV);
  };
  if (RetTy->isRecordType()) {
    llvm::Type *RecTy = CGM.getTypes().ConvertTypeForMem(RetTy);
    ForEachVaryingField(RetTy, [&](const FieldDecl *Field, unsigned Index,
                                   int Loc) {
      llvm::Value *V;
      if (SRetAlloca)
        V = B.CreateLoad(CGM.getTypes().ConvertTypeForMem(Field->getType()),
                         B.CreateStructGEP(RecTy, SRetAlloca, Index));
      else
        V = B.CreateExtractValue(CI, Index);
      if (Loc < 0)
        StoreOutput(V, "wc.position", kBuiltInDecoration, kBuiltInPosition);
      else
        StoreOutput(V, "wc.out." + Field->getName(), kLocationDecoration,
                    (unsigned)Loc);
    });
  } else if (!RetTy->isVoidType()) {
    // Scalar/vector return: fragment color at Location 0 (vertex entries
    // returning a bare [[position]] float4 also land here — BuiltIn instead).
    if (IsVertex)
      StoreOutput(CI, "wc.position", kBuiltInDecoration, kBuiltInPosition);
    else
      StoreOutput(CI, "wc.color", kLocationDecoration, 0);
  }
  B.CreateRetVoid();
}
