//===- PrepareVectorContractLoops.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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
    auto maps = contract.getIndexingMapsArray();
    MLIRContext *ctx = contract.getContext();
    auto m = getAffineDimExpr(0, ctx);
    auto n = getAffineDimExpr(1, ctx);
    auto k = getAffineDimExpr(2, ctx);
    auto par = vector::IteratorType::parallel;
    auto red = vector::IteratorType::reduction;
    if ((isRowMajorMatmul(contract.getIndexingMaps()) &&
         llvm::equal(iterators, ArrayRef{par, par, red}) &&
         maps[0] == AffineMap::get(3, 0, {m, k}, ctx) &&
         maps[1] == AffineMap::get(3, 0, {k, n}, ctx) &&
         maps[2] == AffineMap::get(3, 0, {m, n}, ctx)))
      return contract;

    auto v = getAffineDimExpr(3, ctx);
    if (x86::isInVnniLayout(contract, maps) &&
        llvm::equal(iterators, ArrayRef{par, par, red, red}) &&
        maps[0] == AffineMap::get(4, 0, {m, k, v}, ctx) &&
        maps[1] == AffineMap::get(4, 0, {k, n, v}, ctx) &&
        maps[2] == AffineMap::get(4, 0, {m, n}, ctx))
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

      // scf.for makes overflowing induction-variable values undefined.
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
      // All signed IV values lie between the two bounds.
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

static bool isSimpleReadOutsideLoop(vector::TransferReadOp read,
                                    scf::ForOp loop) {
  return read && read->getParentOp() == loop &&
         isa<MemRefType>(read.getBase().getType()) &&
         loop.isDefinedOutsideOfLoop(read.getBase()) &&
         read.getPermutationMap().isMinorIdentity() && !read.getMask() &&
         !read.hasOutOfBoundsDim();
}

static bool canShiftLowerBound(scf::ForOp loop, unsigned indexWidth) {
  Value lb = loop.getLowerBound(), ub = loop.getUpperBound();
  auto lower = getConstantIntValue(lb);
  auto upper = getConstantIntValue(ub);
  if (lower && upper && *lower >= 0 && *upper >= 0)
    return true;

  return indexWidth > 32 && fitsInSignedWidth(lb, 32) &&
         fitsInSignedWidth(ub, 32);
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
      if (!isSimpleReadOutsideLoop(read, loop))
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
      if (bound.getNumResults() != 1 ||
          llvm::any_of(boundOperands, [&](auto valueDim) {
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
    for (ReadOffset &offset : offsets) {
      if (!shift &&
          offset.read.getIndices()[offset.position] == loop.getInductionVar())
        continue;
      SmallVector<int64_t> strides;
      int64_t baseOffset;
      if (failed(cast<MemRefType>(offset.read.getBase().getType())
                     .getStridesAndOffset(strides, baseOffset)))
        return failure();
    }

    rewriter.setInsertionPoint(loop);
    Location loc = loop.getLoc();
    SmallVector<Value> subviews;
    for (ReadOffset &offset : offsets) {
      if (!shift &&
          offset.read.getIndices()[offset.position] == loop.getInductionVar()) {
        subviews.push_back({});
        continue;
      }
      Value base = offset.read.getBase();
      auto type = cast<MemRefType>(base.getType());
      SmallVector<OpFoldResult> subOffsets, sizes, strides;
      SmallVector<OpFoldResult> mapOperands;
      for (auto valueDim : offset.operands)
        mapOperands.push_back(valueDim.first);
      Value delta = getValueOrCreateConstantIndexOp(
          rewriter, loc,
          affine::makeComposedFoldedAffineApply(rewriter, loc, offset.map,
                                                mapOperands));
      Value first =
          shift ? arith::AddIOp::create(rewriter, loc, lb, delta) : delta;
      Value dim = memref::DimOp::create(rewriter, loc, base, offset.position);
      Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
      // The subview must stay in bounds even when the loop has zero trips.
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
      if (!subviews[i])
        continue;
      ReadOffset &offset = offsets[i];
      rewriter.modifyOpInPlace(offset.read, [&] {
        offset.read.getBaseMutable().assign(subviews[i]);
        SmallVector<Value> indices(offset.read.getIndices());
        indices[offset.position] = iv;
        offset.read.getIndicesMutable().assign(indices);
      });
    }
    return success();
  }
};
struct ExpandedRead {
  vector::TransferReadOp read;
  MemRefType type;
  SmallVector<ReassociationIndices> reassociation;
  int64_t lastIndex;
};

static std::optional<ExpandedRead>
getExpandableRead(vector::TransferReadOp read, unsigned factor,
                  scf::ForOp loop) {
  if (!isSimpleReadOutsideLoop(read, loop))
    return std::nullopt;
  auto type = dyn_cast<MemRefType>(read.getBase().getType());
  if (!type || type.getRank() < 2 || type.isDynamicDim(type.getRank() - 1) ||
      type.getShape().back() % factor != 0)
    return std::nullopt;
  auto lastIndex = getConstantIntValue(read.getIndices().back());
  if (!lastIndex || *lastIndex % factor != 0)
    return std::nullopt;

  SmallVector<int64_t> shape(type.getShape());
  shape.back() /= factor;
  shape.push_back(factor);
  SmallVector<ReassociationIndices> reassociation;
  for (int64_t i = 0; i < type.getRank(); ++i)
    reassociation.push_back(i == type.getRank() - 1
                                ? ReassociationIndices{i, i + 1}
                                : ReassociationIndices{i});
  FailureOr<MemRefType> expanded =
      memref::ExpandShapeOp::computeExpandedType(type, shape, reassociation);
  if (failed(expanded))
    return std::nullopt;
  return ExpandedRead{read, *expanded, reassociation, *lastIndex / factor};
}

static Value buildExpandedRead(PatternRewriter &rewriter, Location loc,
                               ExpandedRead &info, VectorType resultType) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(info.read->getParentOp());
  Value expanded = memref::ExpandShapeOp::create(
      rewriter, loc, info.type, info.read.getBase(), info.reassociation);
  Value lastIndex =
      arith::ConstantIndexOp::create(rewriter, loc, info.lastIndex);
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  rewriter.setInsertionPoint(info.read);
  SmallVector<Value> indices(info.read.getIndices());
  indices.back() = lastIndex;
  indices.push_back(zero);
  auto packed = vector::TransferReadOp::create(
      rewriter, loc, resultType, expanded, indices, info.read.getPadding());
  packed.setInBoundsAttr(
      rewriter.getBoolArrayAttr(SmallVector<bool>(resultType.getRank(), true)));
  return packed;
}

struct FoldVNNIUnpacking : OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contract,
                                PatternRewriter &rewriter) const override {
    auto loop = dyn_cast<scf::ForOp>(contract->getParentOp());
    if (!loop || getCandidate(loop) != contract)
      return failure();

    auto lhsType = contract.getLhsType();
    auto rhsType = contract.getRhsType();
    auto accType = dyn_cast<VectorType>(contract.getAccType());
    Type element = lhsType.getElementType();
    if (!accType || lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
        accType.getRank() != 2 || element != rhsType.getElementType() ||
        !(element.isBF16() || element.isInteger(8)))
      return failure();
    unsigned factor = 32 / element.getIntOrFloatBitWidth();
    if (lhsType.getDimSize(1) != rhsType.getDimSize(0) ||
        lhsType.getDimSize(1) % factor != 0)
      return failure();

    auto shapeCast = contract.getRhs().getDefiningOp<vector::ShapeCastOp>();
    auto transpose =
        shapeCast ? shapeCast.getSource().getDefiningOp<vector::TransposeOp>()
                  : vector::TransposeOp();
    if (!transpose || transpose->getParentOp() != loop)
      return failure();
    VectorType rhsPackedType =
        VectorType::get({lhsType.getDimSize(1) / factor, rhsType.getDimSize(1),
                         int64_t(factor)},
                        element);
    auto sourceType = cast<VectorType>(transpose.getVector().getType());
    if (sourceType.getRank() < rhsPackedType.getRank())
      return failure();
    unsigned leading = sourceType.getRank() - rhsPackedType.getRank();
    if (!llvm::all_of(sourceType.getShape().take_front(leading),
                      [](int64_t dim) { return dim == 1; }) ||
        !llvm::equal(sourceType.getShape().drop_front(leading),
                     rhsPackedType.getShape()))
      return failure();
    for (unsigned i = 0; i < leading + 1; ++i)
      if (transpose.getPermutation()[i] != i)
        return failure();
    if (transpose.getPermutation()[leading + 1] != leading + 2 ||
        transpose.getPermutation()[leading + 2] != leading + 1)
      return failure();

    auto lhsRead = contract.getLhs().getDefiningOp<vector::TransferReadOp>();
    auto lhs = getExpandableRead(lhsRead, factor, loop);
    auto broadcast = transpose.getVector().getDefiningOp<vector::BroadcastOp>();
    Value packedSource =
        broadcast ? broadcast.getSource() : transpose.getVector();
    if (broadcast && packedSource.getType() != rhsPackedType)
      return failure();
    auto packedCast = packedSource.getDefiningOp<vector::ShapeCastOp>();
    auto rhsRead =
        packedCast
            ? packedCast.getSource().getDefiningOp<vector::TransferReadOp>()
            : packedSource.getDefiningOp<vector::TransferReadOp>();
    if (!lhs || !isSimpleReadOutsideLoop(rhsRead, loop))
      return failure();
    std::optional<ExpandedRead> splitRhs;
    if (packedCast) {
      auto flatType = rhsRead.getVectorType();
      if (flatType.getRank() != 2 ||
          flatType.getDimSize(0) != rhsPackedType.getDimSize(0) ||
          flatType.getDimSize(1) % factor != 0 ||
          flatType.getDimSize(1) / factor != rhsPackedType.getDimSize(1) ||
          !(splitRhs = getExpandableRead(rhsRead, factor, loop)))
        return failure();
    } else if (rhsRead.getVectorType() != rhsPackedType ||
               getConstantIntValue(rhsRead.getIndices().back()) != 0) {
      return failure();
    }

    Location loc = contract.getLoc();
    VectorType lhsPackedType =
        VectorType::get({lhsType.getDimSize(0), lhsType.getDimSize(1) / factor,
                         int64_t(factor)},
                        element);
    Value packedLhs = buildExpandedRead(rewriter, loc, *lhs, lhsPackedType);
    Value packedRhs =
        splitRhs ? buildExpandedRead(rewriter, loc, *splitRhs, rhsPackedType)
                 : rhsRead.getResult();
    rewriter.setInsertionPoint(contract);
    auto ctx = rewriter.getContext();
    auto m = rewriter.getAffineDimExpr(0);
    auto n = rewriter.getAffineDimExpr(1);
    auto k = rewriter.getAffineDimExpr(2);
    auto v = rewriter.getAffineDimExpr(3);
    auto vnniMaps =
        rewriter.getAffineMapArrayAttr({AffineMap::get(4, 0, {m, k, v}, ctx),
                                        AffineMap::get(4, 0, {k, n, v}, ctx),
                                        AffineMap::get(4, 0, {m, n}, ctx)});
    auto iterators = rewriter.getArrayAttr(
        {vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
         vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
         vector::IteratorTypeAttr::get(ctx, vector::IteratorType::reduction),
         vector::IteratorTypeAttr::get(ctx, vector::IteratorType::reduction)});
    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        contract, packedLhs, packedRhs, contract.getAcc(), vnniMaps, iterators,
        contract.getKind(), contract.getFastmath());
    return success();
  }
};
} // namespace

void x86::populatePrepareVectorContractLoopsPatterns(
    RewritePatternSet &patterns) {
  patterns.add<NormalizeCounter, RebaseReadIndices, FoldVNNIUnpacking>(
      patterns.getContext());
}