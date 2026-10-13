//===- PrepareVectorContractLoops.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/X86/Transforms.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

void x86::populatePrepareVectorContractLoopsPatterns(
    RewritePatternSet &patterns) {}