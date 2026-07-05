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
//    capability, an optional Vulkan feature).
//
//===----------------------------------------------------------------------===//

#include "CGMetalRuntime.h"
#include "CGDebugInfo.h"
#include "CodeGenFunction.h"
#include "CodeGenModule.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
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

llvm::Value *CGMetalRuntime::emitBufferElementPtr(CodeGenFunction &CGF,
                                                  const ParmVarDecl *PD,
                                                  llvm::Value *Idx,
                                                  QualType ElemTy) {
  const auto *FD = dyn_cast_or_null<FunctionDecl>(CGF.CurFuncDecl);
  assert(FD && "buffer parameter outside a Metal entry function");
  auto *BindingAttr = PD->getAttr<MetalBufferBindingAttr>();
  unsigned Set = stageDescriptorSet(FD);
  unsigned Binding = BindingAttr->getIndex();

  llvm::Type *ElemLLVMTy = CGF.ConvertTypeForMem(ElemTy);
  auto *RuntimeArrTy = llvm::ArrayType::get(ElemLLVMTy, 0);
  auto *HandleTy = llvm::TargetExtType::get(
      CGM.getLLVMContext(), "spirv.VulkanBuffer", {RuntimeArrTy},
      {kStorageBufferSC, /*Writable=*/0});

  llvm::Constant *&NameStr = BufferNameStrs[PD];
  if (!NameStr)
    NameStr = CGM.GetAddrOfConstantCString(PD->getNameAsString(), ".str.wcbuf")
                  .getPointer();

  llvm::Function *HandleFn = CGM.getIntrinsic(
      llvm::Intrinsic::spv_resource_handlefrombinding, {HandleTy});
  llvm::Value *Handle = CGF.Builder.CreateCall(
      HandleFn, {CGF.Builder.getInt32(Set), CGF.Builder.getInt32(Binding),
                 CGF.Builder.getInt32(1), CGF.Builder.getInt32(0),
                 CGF.Builder.getInt1(false), NameStr});

  llvm::Value *Idx32 =
      CGF.Builder.CreateZExtOrTrunc(Idx, CGF.Builder.getInt32Ty());
  auto *PtrTy = llvm::PointerType::get(CGM.getLLVMContext(), 11);
  llvm::Function *GetPtrFn = CGM.getIntrinsic(
      llvm::Intrinsic::spv_resource_getpointer, {PtrTy, HandleTy});
  return CGF.Builder.CreateCall(GetPtrFn, {Handle, Idx32});
}

LValue CGMetalRuntime::emitBufferSubscriptLValue(CodeGenFunction &CGF,
                                                 const ArraySubscriptExpr *E,
                                                 const ParmVarDecl *PD) {
  llvm::Value *Idx = CGF.EmitScalarExpr(E->getIdx());
  QualType ElemTy = E->getType();
  llvm::Value *Ptr = emitBufferElementPtr(CGF, PD, Idx, ElemTy);
  CharUnits Align = CGF.getContext().getTypeAlignInChars(ElemTy);
  return CGF.MakeAddrLValue(
      Address(Ptr, CGF.ConvertTypeForMem(ElemTy), Align), ElemTy);
}

LValue CGMetalRuntime::emitBufferParamDeclRefLValue(CodeGenFunction &CGF,
                                                    const ParmVarDecl *PD) {
  QualType T = PD->getType();
  if (const auto *RefTy = T->getAs<ReferenceType>()) {
    // constant T& u [[buffer(N)]] — element 0 of the SSBO.
    QualType ElemTy = RefTy->getPointeeType();
    llvm::Value *Ptr =
        emitBufferElementPtr(CGF, PD, CGF.Builder.getInt32(0), ElemTy);
    CharUnits Align = CGF.getContext().getTypeAlignInChars(ElemTy);
    return CGF.MakeAddrLValue(
        Address(Ptr, CGF.ConvertTypeForMem(ElemTy), Align), ElemTy);
  }
  // A raw pointer use other than a direct subscript (address taken, passed
  // to a helper, pointer arithmetic) is not representable in this slice.
  CGM.Error(PD->getLocation(),
            "Metal buffer pointer parameters only support direct subscript "
            "access in this implementation");
  llvm::Type *Ty = CGF.ConvertType(T);
  CharUnits Align = CGF.getContext().getTypeAlignInChars(T);
  llvm::Value *Poison = llvm::PoisonValue::get(CGM.UnqualPtrTy);
  return CGF.MakeAddrLValue(Address(Poison, Ty, Align), T);
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
  if (CGM.getCodeGenOpts().OptimizationLevel == 0)
    EntryFn->addFnAttr(llvm::Attribute::OptimizeNone);
  EntryFn->addFnAttr(llvm::Attribute::NoInline);

  // The user function must fold into the wrapper: logical SPIR-V has no
  // linkage, and the poison placeholder args below must vanish.
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
    if (PD->hasAttr<MetalBufferBindingAttr>()) {
      // Never read: every use inside the user function is lowered through
      // llvm.spv.resource handles (see emitBufferSubscriptLValue).
      Args.emplace_back(PoisonValue::get(Param.getType()));
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
              "[[buffer(N)]] or [[stage_in]]");
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
