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
#include "llvm/TargetParser/Triple.h"

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

unsigned CGMetalRuntime::stageDescriptorSet(const FunctionDecl *FD) {
  // <Metal/MTLLibrary.h>: descriptor set 0 = vertex stage, 1 = fragment.
  return FD->hasAttr<MetalVertexAttr>() ? 0u : 1u;
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
      {kStorageBufferSC, /*Writable=*/0});

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

  if (Name == "get_width" && E->getNumArgs() == 0) {
    // OpImageQuerySizeLod at level 0 -> (width, height); component 0.
    auto *SizeTy = llvm::FixedVectorType::get(CGF.Builder.getInt32Ty(), 2);
    llvm::Function *QueryFn =
        CGM.getIntrinsic(llvm::Intrinsic::spv_resource_imagequerysizelod,
                         {SizeTy, Img->getType()});
    llvm::Value *Size =
        CGF.Builder.CreateCall(QueryFn, {Img, CGF.Builder.getInt32(0)});
    return RValue::get(CGF.Builder.CreateExtractElement(
        Size, CGF.Builder.getInt32(0), "wc.tex.width"));
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
                         IsVertex ? llvm::Triple::Vertex : llvm::Triple::Pixel));
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
    CGM.Error(PD->getLocation(),
              "unsupported Metal entry parameter: expected [[vertex_id]], "
              "[[position]], [[buffer(N)]], [[texture(N)]], [[sampler(N)]] "
              "or [[stage_in]]");
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
