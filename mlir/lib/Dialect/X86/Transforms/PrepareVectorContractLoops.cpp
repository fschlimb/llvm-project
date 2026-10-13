//===- PrepareVectorContractLoops.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/X86/Transforms.h"
#include "mlir/Dialect/X86/Utils/X86Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"

using namespace mlir;

namespace {
static vector::ContractionOp getCandidate(scf::ForOp loop) {
  if (loop.getNumRegionIterArgs() != 1)
    return {};

  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  for (Operation &op : loop.getBody()->without_terminator()) {
    auto contract = dyn_cast<vector::ContractionOp>(&op);
    if (!contract || contract.getKind() != vector::CombiningKind::ADD ||
        contract.getAcc() != loop.getRegionIterArgs().front() ||
        !contract.getAcc().hasOneUse() || !contract.getResult().hasOneUse() ||
        yield->getNumOperands() != 1 ||
        yield->getOperand(0) != contract.getResult())
      continue;

    auto iterators = contract.getIteratorTypesArray();
    auto par = vector::IteratorType::parallel;
    auto red = vector::IteratorType::reduction;
    if ((isRowMajorMatmul(contract.getIndexingMaps()) &&
         llvm::equal(iterators, ArrayRef{par, par, red})) ||
        (x86::isInVnniLayout(contract, contract.getIndexingMapsArray()) &&
         llvm::equal(iterators, ArrayRef{par, par, red, red})))
      return contract;
  }
  return {};
}

static bool fitsInSignedWidth(Value value, unsigned width) {
  if (auto constant = getConstantIntValue(value))
    return llvm::isIntN(width, *constant);
  auto cast = value.getDefiningOp<arith::IndexCastOp>();
  return cast && isa<IntegerType>(cast.getIn().getType()) &&
         cast.getIn().getType().getIntOrFloatBitWidth() <= width;
}

struct NormalizeCounter : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp loop,
                                PatternRewriter &rewriter) const override {
    if (loop.getUnsignedCmp() || !getCandidate(loop))
      return failure();

    Value iv = loop.getInductionVar();
    if (auto intType = dyn_cast<IntegerType>(iv.getType())) {
      unsigned indexWidth =
          DataLayout::closest(loop).getTypeSizeInBits(rewriter.getIndexType());
      if (intType.getWidth() > indexWidth)
        return failure();

      SmallVector<arith::IndexCastOp> indexUsers;
      for (Operation *user : iv.getUsers())
        if (auto cast = dyn_cast<arith::IndexCastOp>(user);
            cast && cast.getType().isIndex())
          indexUsers.push_back(cast);
      if (indexUsers.empty())
        return failure();

      rewriter.setInsertionPoint(loop);
      Value lb = arith::IndexCastOp::create(rewriter, loop.getLoc(),
                                            rewriter.getIndexType(),
                                            loop.getLowerBound());
      Value ub = arith::IndexCastOp::create(rewriter, loop.getLoc(),
                                            rewriter.getIndexType(),
                                            loop.getUpperBound());
      Value step = arith::IndexCastOp::create(
          rewriter, loop.getLoc(), rewriter.getIndexType(), loop.getStep());
      rewriter.modifyOpInPlace(loop, [&] {
        loop.setLowerBound(lb);
        loop.setUpperBound(ub);
        loop.setStep(step);
        cast<BlockArgument>(iv).setType(rewriter.getIndexType());
      });
      rewriter.setInsertionPointToStart(loop.getBody());
      Value oldIV =
          arith::IndexCastOp::create(rewriter, loop.getLoc(), intType, iv);
      rewriter.replaceAllUsesExcept(iv, oldIV, oldIV.getDefiningOp());
      for (arith::IndexCastOp cast : indexUsers)
        rewriter.replaceOp(cast, iv);
      return success();
    }

    for (Operation *user : iv.getUsers()) {
      auto inner = dyn_cast<arith::IndexCastOp>(user);
      if (!inner || !isa<IntegerType>(inner.getType()))
        continue;
      unsigned width = cast<IntegerType>(inner.getType()).getWidth();
      if (!fitsInSignedWidth(loop.getLowerBound(), width) ||
          !fitsInSignedWidth(loop.getUpperBound(), width))
        continue;
      for (Operation *castUser : inner.getResult().getUsers()) {
        auto outer = dyn_cast<arith::IndexCastOp>(castUser);
        if (outer && outer.getType().isIndex()) {
          rewriter.replaceOp(outer, iv);
          return success();
        }
      }
    }
    return failure();
  }
};
} // namespace

void x86::populatePrepareVectorContractLoopsPatterns(
    RewritePatternSet &patterns) {
  patterns.add<NormalizeCounter>(patterns.getContext());
}