// A header that belongs to no module (glibc's <stdio.h> on Linux) is parsed
// again into every module that includes it. A swift_attr added by a
// redeclaration in one module must survive the merge with another module's
// copy, whichever of them is loaded last: the Swift importer reads attributes
// from the most recent declaration.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   -I%t %t/annotated-first.c -ast-dump-all -ast-dump-filter stream \
// RUN:   | FileCheck %s --check-prefix=FIRST
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache \
// RUN:   -I%t %t/plain-first.c -ast-dump-all -ast-dump-filter stream \
// RUN:   | FileCheck %s --check-prefix=LAST

//--- module.modulemap
module Annotated {
  header "annotated.h"
  export *
}
module Plain {
  header "plain.h"
  export *
}

//--- stream.h
extern int *stream;

//--- annotated.h
#include "stream.h"
extern int *stream __attribute__((swift_attr("nonisolated(unsafe)")));

//--- plain.h
#include "stream.h"

//--- annotated-first.c
#include "annotated.h"
#include "plain.h"
int *use(void) { return stream; }

// Plain's copy is loaded last, and inherits the attribute.
// FIRST: VarDecl {{.*}} <{{.*}}stream.h:1:1, col:13> col:13 imported in Annotated {{.*}}stream 'int *' extern
// FIRST: VarDecl {{.*}} prev {{.*}} <{{.*}}annotated.h:2:1, col:13> col:13 imported in Annotated {{.*}}stream 'int *' extern
// FIRST-NEXT: SwiftAttrAttr {{.*}} "nonisolated(unsafe)"
// FIRST: VarDecl {{.*}} prev {{.*}} <{{.*}}stream.h:1:1, col:13> col:13 imported in Plain {{.*}}stream 'int *' extern
// FIRST-NEXT: SwiftAttrAttr {{.*}} Inherited "nonisolated(unsafe)"

//--- plain-first.c
#include "plain.h"
#include "annotated.h"
int *use(void) { return stream; }

// Annotated's redeclaration is loaded last and carries its own attribute.
// LAST: VarDecl {{.*}} <{{.*}}stream.h:1:1, col:13> col:13 imported in Plain {{.*}}stream 'int *' extern
// LAST: VarDecl {{.*}} prev {{.*}} <{{.*}}annotated.h:2:1, col:13> col:13 imported in Annotated {{.*}}stream 'int *' extern
// LAST-NEXT: SwiftAttrAttr {{.*}} "nonisolated(unsafe)"
