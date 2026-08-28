// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -fobjc-runtime=gnustep-2.0 -fexceptions -fobjc-exceptions -Wno-objc-root-class -emit-llvm -o - %s | FileCheck %s
//
// Under MSVC (funclet) EH, an ObjC @finally is outlined as an SEH helper
// function, and enclosing-scope locals it reads must be captured via
// llvm.localescape/localrecover. ObjC 'self' is an ImplicitParamDecl, which
// the SEH CaptureFinder used to skip, so a @finally touching an ivar died
// with "DeclRefExpr for Decl not entered in LocalDeclMap" at IR generation.

@interface Archiver {
  id _enc;
  unsigned _keyNum;
}
- (void)encodeObject:(id)anObject with:(id)m;
- (void)encodeWithCoder:(id)coder;
@end

@implementation Archiver
- (void)encodeObject:(id)anObject with:(id)m {
  id savedEnc = _enc;
  unsigned savedKeyNum = _keyNum;
  _enc = m;
  _keyNum = 0;
  @try {
    [anObject encodeWithCoder:(id)self];
  } @finally {
    // Reads two enclosing locals and writes two ivars (through 'self').
    _keyNum = savedKeyNum;
    _enc = savedEnc;
  }
}
- (void)encodeWithCoder:(id)coder {}
@end

// The parent frame escapes the captured allocas...
// CHECK-LABEL: define {{.*}}encodeObject
// CHECK: call void (...) @llvm.localescape(

// ...and the outlined finally funclet recovers 'self', 'savedEnc', and
// 'savedKeyNum' from the parent frame instead of crashing on them.
// CHECK: define internal void @"?fin$
// CHECK: call ptr @llvm.localrecover(
// CHECK: call ptr @llvm.localrecover(
// CHECK: call ptr @llvm.localrecover(
