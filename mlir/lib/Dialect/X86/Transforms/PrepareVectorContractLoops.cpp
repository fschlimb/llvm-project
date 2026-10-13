//===- PrepareVectorContractLoops.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/X86/Transforms.h"
#include "mlir/Dialect/X86/Utils/X86Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

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

struct ReadOffset {
  vector::TransferReadOp read;
  unsigned position;
  AffineMap map;
  ValueDimList operands;
};

static bool canShiftLowerBound(scf::ForOp loop, unsigned indexWidth) {
  Value lb = loop.getLowerBound(), ub = loop.getUpperBound();
  auto lower = getConstantIntValue(lb);
  auto upper = getConstantIntValue(ub);
  if (lower && upper && *lower >= 0 && *upper >= 0)
    return true;

  auto fromNarrowInt = [](Value value) {
    auto cast = value.getDefiningOp<arith::IndexCastOp>();
    return cast && isa<IntegerType>(cast.getIn().getType()) &&
           cast.getIn().getType().getIntOrFloatBitWidth() <= 32;
  };
  return indexWidth > 32 &&
         (fromNarrowInt(lb) || (lower && llvm::isIntN(32, *lower))) &&
         (fromNarrowInt(ub) || (upper && llvm::isIntN(32, *upper)));
}

struct RebaseReadIndices : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp loop,
                                PatternRewriter &rewriter) const override {
    auto contract = getCandidate(loop);
    if (!contract || loop.getUnsignedCmp() ||
        !loop.getInductionVar().getType().isIndex())
      return failure();

    SmallVector<ReadOffset> offsets;
    bool needsRebasing = false;
    for (Value operand : {contract.getLhs(), contract.getRhs()}) {
      auto read = operand.getDefiningOp<vector::TransferReadOp>();
      if (!read || read->getParentOp() != loop ||
          !isa<MemRefType>(read.getBase().getType()) ||
          !loop.isDefinedOutsideOfLoop(read.getBase()) ||
          !read.getPermutationMap().isMinorIdentity() || read.getMask() ||
          read.hasOutOfBoundsDim())
        return failure();

      std::optional<unsigned> position;
      for (auto [i, index] : llvm::enumerate(read.getIndices())) {
        if (!loop.isDefinedOutsideOfLoop(index)) {
          if (position)
            return failure();
          position = i;
        }
      }
      if (!position)
        return failure();

      Value index = read.getIndices()[*position];
      needsRebasing |= index != loop.getInductionVar();
      MLIRContext *ctx = rewriter.getContext();
      auto map = AffineMap::get(
          0, 2, getAffineSymbolExpr(0, ctx) - getAffineSymbolExpr(1, ctx));
      ValueBoundsConstraintSet::Variable delta(
          map, ValueRange{index, loop.getInductionVar()});
      AffineMap bound;
      ValueDimList boundOperands;
      auto outside = [&](Value value, std::optional<int64_t>,
                         ValueBoundsConstraintSet &) {
        return loop.isDefinedOutsideOfLoop(value);
      };
      if (failed(ValueBoundsConstraintSet::computeBound(
              bound, boundOperands, presburger::BoundType::EQ, delta, outside)))
        return failure();
      if (llvm::any_of(boundOperands, [&](auto valueDim) {
            return valueDim.second ||
                   !loop.isDefinedOutsideOfLoop(valueDim.first);
          }))
        return failure();
      offsets.push_back({read, *position, bound, std::move(boundOperands)});
    }
    if (!needsRebasing)
      return failure();

    Value lb = loop.getLowerBound();
    bool shift = getConstantIntValue(lb) != 0;
    if (shift &&
        !canShiftLowerBound(loop, DataLayout::closest(loop).getTypeSizeInBits(
                                      rewriter.getIndexType())))
      return failure();

    rewriter.setInsertionPoint(loop);
    Location loc = loop.getLoc();
    SmallVector<Value> subviews;
    for (ReadOffset &offset : offsets) {
      Value base = offset.read.getBase();
      auto type = cast<MemRefType>(base.getType());
      SmallVector<OpFoldResult> subOffsets, sizes, strides;
      Value delta = getValueOrCreateConstantIndexOp(
          rewriter, loc,
          *arith::reifyValueBound(
              rewriter, loc, presburger::BoundType::EQ,
              ValueBoundsConstraintSet::Variable(
                  offset.map,
                  llvm::map_to_vector(offset.operands,
                                      [](auto pair) { return pair.first; })),
              [&](Value value, std::optional<int64_t>,
                  ValueBoundsConstraintSet &) {
                return loop.isDefinedOutsideOfLoop(value);
              }));
      Value first =
          shift ? arith::AddIOp::create(rewriter, loc, lb, delta) : delta;
      Value dim = memref::DimOp::create(rewriter, loc, base, offset.position);
      Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
      Value clamped = arith::MinSIOp::create(
          rewriter, loc, arith::MaxSIOp::create(rewriter, loc, first, zero),
          dim);
      for (unsigned i = 0; i < type.getRank(); ++i) {
        strides.push_back(rewriter.getIndexAttr(1));
        if (i == offset.position) {
          subOffsets.push_back(clamped);
          sizes.push_back(
              arith::SubIOp::create(rewriter, loc, dim, clamped).getResult());
        } else {
          subOffsets.push_back(rewriter.getIndexAttr(0));
          sizes.push_back(
              type.isDynamicDim(i)
                  ? OpFoldResult(memref::DimOp::create(rewriter, loc, base, i)
                                     .getResult())
                  : OpFoldResult(rewriter.getIndexAttr(type.getDimSize(i))));
        }
      }
      subviews.push_back(memref::SubViewOp::create(rewriter, loc, base,
                                                   subOffsets, sizes, strides));
    }

    Value iv = loop.getInductionVar();
    if (shift) {
      Value newUb =
          arith::SubIOp::create(rewriter, loc, loop.getUpperBound(), lb);
      Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
      rewriter.modifyOpInPlace(loop, [&] {
        loop.setLowerBound(zero);
        loop.setUpperBound(newUb);
      });
      rewriter.setInsertionPointToStart(loop.getBody());
      denormalizeInductionVariable(rewriter, loc, iv, lb,
                                   rewriter.getIndexAttr(1));
    }

    for (unsigned i = 0; i < offsets.size(); ++i) {
      ReadOffset &offset = offsets[i];
      rewriter.modifyOpInPlace(offset.read, [&] {
        offset.read->setOperand(0, subviews[i]);
        offset.read->setOperand(1 + offset.position, iv);
      });
    }
    return success();
  }
};
} // namespace

void x86::populatePrepareVectorContractLoopsPatterns(
    RewritePatternSet &patterns) {
  patterns.add<NormalizeCounter, RebaseReadIndices>(patterns.getContext());
}