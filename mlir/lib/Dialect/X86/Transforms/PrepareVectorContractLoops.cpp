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
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/X86/Transforms.h"
#include "mlir/Dialect/X86/Utils/X86Utils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "x86-prepare-vector-contract-loops"

using namespace mlir;

namespace {
vector::ContractionOp getCandidate(scf::ForOp loop) {
  if (loop.getNumRegionIterArgs() != 1) {
    LLVM_DEBUG(llvm::dbgs() << "reject loop at " << loop.getLoc()
                            << ": expected one iter_arg\n");
    return {};
  }

  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  for (Operation &op : loop.getBody()->without_terminator()) {
    auto contract = dyn_cast<vector::ContractionOp>(&op);
    if (!contract)
      continue;
    if (contract.getKind() != vector::CombiningKind::ADD ||
        contract.getAcc() != loop.getRegionIterArgs().front() ||
        !contract.getAcc().hasOneUse() || !contract.getResult().hasOneUse() ||
        yield->getNumOperands() != 1 ||
        yield->getOperand(0) != contract.getResult()) {
      LLVM_DEBUG(llvm::dbgs() << "reject contraction at " << contract.getLoc()
                              << ": not a single loop-carried accumulation\n");
      continue;
    }

    auto iterators = contract.getIteratorTypesArray();
    auto maps = contract.getIndexingMapsArray();
    MLIRContext *ctx = contract.getContext();
    auto m = getAffineDimExpr(0, ctx);
    auto n = getAffineDimExpr(1, ctx);
    auto k = getAffineDimExpr(2, ctx);
    auto par = vector::IteratorType::parallel;
    auto red = vector::IteratorType::reduction;
    if (llvm::equal(iterators, ArrayRef{par, par, red}) &&
        maps[0] == AffineMap::get(3, 0, {m, k}, ctx) &&
        maps[1] == AffineMap::get(3, 0, {k, n}, ctx) &&
        maps[2] == AffineMap::get(3, 0, {m, n}, ctx))
      return contract;

    auto v = getAffineDimExpr(3, ctx);
    if (x86::isInVnniLayout(contract, maps) &&
        llvm::equal(iterators, ArrayRef{par, par, red, red}) &&
        maps[0] == AffineMap::get(4, 0, {m, k, v}, ctx) &&
        maps[1] == AffineMap::get(4, 0, {k, n, v}, ctx) &&
        maps[2] == AffineMap::get(4, 0, {m, n}, ctx))
      return contract;
    LLVM_DEBUG(llvm::dbgs() << "reject contraction at " << contract.getLoc()
                            << ": noncanonical matmul layout or iterators\n");
  }
  return {};
}

bool fitsInSignedWidth(Value value, unsigned width) {
  if (auto constant = getConstantIntValue(value))
    return llvm::isIntN(width, *constant);
  auto cast = value.getDefiningOp<arith::IndexCastOp>();
  return cast && isa<IntegerType>(cast.getIn().getType()) &&
         cast.getIn().getType().getIntOrFloatBitWidth() <= width;
}

/// Returns true if every value that `value` takes is a multiple of `divisor`.
bool isMultipleOf(Value value, const APInt &divisor) {
  APInt constant;
  if (matchPattern(value, m_ConstantInt(&constant)))
    return constant.srem(divisor).isZero();
  // Signed scf.for IVs take the values lb + j * step without overflow.
  auto loop = scf::getForInductionVarOwner(value);
  return loop && loop.getInductionVar() == value && !loop.getUnsignedCmp() &&
         isMultipleOf(loop.getLowerBound(), divisor) &&
         isMultipleOf(loop.getStep(), divisor);
}

// Expose the IV as a direct read index by folding exact round-downs and
// normalizing safe integer/index conversions.
struct CanonicalizeLoopIV : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp loop,
                                PatternRewriter &rewriter) const override {
    if (loop.getUnsignedCmp())
      return rewriter.notifyMatchFailure(loop, "unsigned loop");
    if (!getCandidate(loop))
      return rewriter.notifyMatchFailure(loop, "no canonical contraction");

    Value iv = loop.getInductionVar();
    // (iv / c) * c is iv when every IV value is a multiple of c.
    SmallVector<Operation *> roundings;
    for (Operation *div : iv.getUsers()) {
      APInt divisor;
      if (!isa<arith::DivSIOp, arith::CeilDivSIOp, arith::FloorDivSIOp>(div) ||
          div->getOperand(0) != iv ||
          !matchPattern(div->getOperand(1), m_ConstantInt(&divisor)) ||
          !divisor.isStrictlyPositive() || !isMultipleOf(iv, divisor))
        continue;
      for (Operation *mul : div->getUsers()) {
        APInt factor;
        if (isa<arith::MulIOp>(mul) &&
            (matchPattern(mul->getOperand(0), m_ConstantInt(&factor)) ||
             matchPattern(mul->getOperand(1), m_ConstantInt(&factor))) &&
            factor == divisor)
          roundings.push_back(mul);
      }
    }
    for (Operation *mul : roundings)
      rewriter.replaceOp(mul, iv);
    if (!roundings.empty())
      return success();

    if (auto intType = dyn_cast<IntegerType>(iv.getType())) {
      unsigned indexWidth =
          DataLayout::closest(loop).getTypeSizeInBits(rewriter.getIndexType());
      if (intType.getWidth() > indexWidth)
        return rewriter.notifyMatchFailure(loop, "integer IV wider than index");

      SmallVector<arith::IndexCastOp> indexUsers;
      for (Operation *user : iv.getUsers())
        if (auto cast = dyn_cast<arith::IndexCastOp>(user);
            cast && cast.getType().isIndex())
          indexUsers.push_back(cast);
      if (indexUsers.empty())
        return rewriter.notifyMatchFailure(
            loop, "integer IV has no index cast users");

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

    bool hasIntegerCast = false, hasSafeIntegerCast = false;
    for (Operation *user : iv.getUsers()) {
      auto inner = dyn_cast<arith::IndexCastOp>(user);
      if (!inner || !isa<IntegerType>(inner.getType()))
        continue;
      hasIntegerCast = true;
      unsigned width = cast<IntegerType>(inner.getType()).getWidth();
      if (!fitsInSignedWidth(loop.getLowerBound(), width) ||
          !fitsInSignedWidth(loop.getUpperBound(), width)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "skip index cast round trip at " << loop.getLoc()
                   << ": bounds do not fit i" << width << '\n');
        continue;
      }
      hasSafeIntegerCast = true;
      // All signed IV values lie between the two bounds.
      for (Operation *castUser : inner.getResult().getUsers()) {
        auto outer = dyn_cast<arith::IndexCastOp>(castUser);
        if (outer && outer.getType().isIndex()) {
          rewriter.replaceOp(outer, iv);
          return success();
        }
      }
    }
    if (!hasIntegerCast)
      return rewriter.notifyMatchFailure(loop, "IV has no cast to integer");
    if (!hasSafeIntegerCast)
      return rewriter.notifyMatchFailure(loop,
                                         "loop bounds do not fit cast width");
    return rewriter.notifyMatchFailure(
        loop, "no cast back to index from a safe cast");
  }
};

struct ReadOffset {
  vector::TransferReadOp read;
  unsigned position;
  AffineMap map;
  ValueDimList operands;
};

bool isSimpleReadOutsideLoop(vector::TransferReadOp read, scf::ForOp loop) {
  return read && read->getParentOp() == loop &&
         isa<MemRefType>(read.getBase().getType()) &&
         loop.isDefinedOutsideOfLoop(read.getBase()) &&
         read.getPermutationMap().isMinorIdentity() && !read.getMask() &&
         !read.hasOutOfBoundsDim();
}

bool canShiftLowerBound(scf::ForOp loop, unsigned indexWidth) {
  Value lb = loop.getLowerBound(), ub = loop.getUpperBound();
  auto lower = getConstantIntValue(lb);
  auto upper = getConstantIntValue(ub);
  if (lower && upper && *lower >= 0 && *upper >= 0)
    return true;

  return indexWidth > 32 && fitsInSignedWidth(lb, 32) &&
         fitsInSignedWidth(ub, 32);
}

/// Returns true if `value` combines the IV and loop invariants using only
/// addi, subi and muli. Unlike divisions, these stay congruent modulo 2^n to
/// the overflow-free arithmetic that value bounds reason about.
bool isRingExpression(Value value, scf::ForOp loop) {
  SmallVector<Value> worklist{value};
  DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value next = worklist.pop_back_val();
    if (!visited.insert(next).second || next == loop.getInductionVar() ||
        loop.isDefinedOutsideOfLoop(next) || matchPattern(next, m_Constant()))
      continue;
    Operation *op = next.getDefiningOp();
    if (!isa_and_present<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op))
      return false;
    llvm::append_range(worklist, op->getOperands());
  }
  return true;
}

// Rebase contraction reads onto the loop IV:
//   - Prove each read index differs from the IV by a loop-invariant offset.
//   - Move that offset into a subview; shift a nonzero loop lower bound to
//   zero.
// For LB = 0, A[m, k + off] becomes a read at [m, k] from a subview at [0,
// off].
struct RebaseReadIndices : OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp loop,
                                PatternRewriter &rewriter) const override {
    auto contract = getCandidate(loop);
    if (!contract)
      return rewriter.notifyMatchFailure(loop, "no canonical contraction");
    if (loop.getUnsignedCmp())
      return rewriter.notifyMatchFailure(loop, "unsigned loop");
    if (!loop.getInductionVar().getType().isIndex())
      return rewriter.notifyMatchFailure(loop, "IV is not index-typed");

    SmallVector<ReadOffset> offsets;
    bool needsRebasing = false;
    for (Value operand : {contract.getLhs(), contract.getRhs()}) {
      auto read = operand.getDefiningOp<vector::TransferReadOp>();
      if (!isSimpleReadOutsideLoop(read, loop))
        return rewriter.notifyMatchFailure(
            loop, "operand is not a simple read from an invariant memref");

      std::optional<unsigned> position;
      for (auto [i, index] : llvm::enumerate(read.getIndices())) {
        if (!loop.isDefinedOutsideOfLoop(index)) {
          if (position)
            return rewriter.notifyMatchFailure(
                loop, "read has multiple loop-dependent indices");
          position = i;
        }
      }
      if (!position)
        return rewriter.notifyMatchFailure(loop,
                                           "read has no loop-dependent index");

      Value index = read.getIndices()[*position];
      if (!isRingExpression(index, loop))
        return rewriter.notifyMatchFailure(
            loop, "read index is not an IV-based add/sub/mul expression");
      needsRebasing |= index != loop.getInductionVar();
      // Represent index - IV so ValueBounds can prove a loop-invariant offset.
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
        return rewriter.notifyMatchFailure(
            loop, "cannot prove an IV-independent read offset");
      if (bound.getNumResults() != 1)
        return rewriter.notifyMatchFailure(loop, "read offset is not singular");
      if (llvm::any_of(boundOperands, [&](auto valueDim) {
            return valueDim.second ||
                   !loop.isDefinedOutsideOfLoop(valueDim.first);
          }))
        return rewriter.notifyMatchFailure(
            loop, "read offset depends on a loop value or dimension");
      offsets.push_back({read, *position, bound, std::move(boundOperands)});
    }
    if (!needsRebasing)
      return rewriter.notifyMatchFailure(loop, "reads already use the IV");

    Value lb = loop.getLowerBound();
    bool shift = getConstantIntValue(lb) != 0;
    if (shift &&
        !canShiftLowerBound(loop, DataLayout::closest(loop).getTypeSizeInBits(
                                      rewriter.getIndexType())))
      return rewriter.notifyMatchFailure(
          loop, "cannot safely shift the loop lower bound");
    for (ReadOffset &offset : offsets) {
      if (!shift &&
          offset.read.getIndices()[offset.position] == loop.getInductionVar())
        continue;
      SmallVector<int64_t> strides;
      int64_t baseOffset;
      if (failed(cast<MemRefType>(offset.read.getBase().getType())
                     .getStridesAndOffset(strides, baseOffset)))
        return rewriter.notifyMatchFailure(loop, "read base is not strided");
    }

    rewriter.setInsertionPoint(loop);
    Location loc = loop.getLoc();
    SmallVector<Value> subviews;
    for (ReadOffset &offset : offsets) {
      // A read already using the unshifted IV needs no subview.
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
      // Materialize the proved offset before the loop.
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
      // Slice the dependent dimension's tail; keep other dimensions whole.
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

    // modify affected reads
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

std::optional<ExpandedRead> getExpandableRead(vector::TransferReadOp read,
                                              unsigned factor,
                                              scf::ForOp loop) {
  if (!isSimpleReadOutsideLoop(read, loop))
    return std::nullopt;
  auto type = cast<MemRefType>(read.getBase().getType());
  if (type.getElementType() != read.getVectorType().getElementType() ||
      type.getRank() < 2 || type.isDynamicDim(type.getRank() - 1) ||
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

Value buildExpandedRead(PatternRewriter &rewriter, Location loc,
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

// Fold RHS unpacking into a VNNI contraction:
//   - Match a packed RHS read transposed and flattened for a flat matmul.
//   - Split the LHS memref's K dimension and reread it in packed lanes.
//   - Contract the packed reads directly over K and the VNNI lane dimension.
// For bf16, 32x32 LHS and RHS become 32x16x2 LHS and 16x32x2 RHS.
struct FoldVNNIUnpacking : OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contract,
                                PatternRewriter &rewriter) const override {
    auto loop = dyn_cast<scf::ForOp>(contract->getParentOp());
    if (!loop)
      return rewriter.notifyMatchFailure(contract,
                                         "not directly in a for loop");
    if (getCandidate(loop) != contract)
      return rewriter.notifyMatchFailure(
          contract, "not the loop's canonical contraction");

    auto lhsType = contract.getLhsType();
    auto rhsType = contract.getRhsType();
    auto accType = dyn_cast<VectorType>(contract.getAccType());
    Type element = lhsType.getElementType();
    if (!accType || lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
        accType.getRank() != 2)
      return rewriter.notifyMatchFailure(
          contract, "expected 2D operands and accumulator");
    if (lhsType.isScalable() || rhsType.isScalable())
      return rewriter.notifyMatchFailure(contract, "scalable operand vectors");
    if (element != rhsType.getElementType())
      return rewriter.notifyMatchFailure(contract,
                                         "operand element types differ");
    if (!(element.isBF16() || element.isInteger(8)))
      return rewriter.notifyMatchFailure(contract,
                                         "expected bf16 or i8 operands");
    unsigned factor = 32 / element.getIntOrFloatBitWidth();
    if (lhsType.getDimSize(1) != rhsType.getDimSize(0))
      return rewriter.notifyMatchFailure(contract, "K dimensions differ");
    if (lhsType.getDimSize(1) % factor != 0)
      return rewriter.notifyMatchFailure(
          contract, "K dimension is not divisible by VNNI factor");

    auto shapeCast = contract.getRhs().getDefiningOp<vector::ShapeCastOp>();
    auto transpose =
        shapeCast ? shapeCast.getSource().getDefiningOp<vector::TransposeOp>()
                  : vector::TransposeOp();
    if (!transpose || transpose->getParentOp() != loop)
      return rewriter.notifyMatchFailure(
          contract, "RHS is not an in-loop transpose followed by shape_cast");
    VectorType rhsPackedType =
        VectorType::get({lhsType.getDimSize(1) / factor, rhsType.getDimSize(1),
                         int64_t(factor)},
                        element);
    auto sourceType = cast<VectorType>(transpose.getVector().getType());
    if (sourceType.getRank() < rhsPackedType.getRank())
      return rewriter.notifyMatchFailure(contract,
                                         "RHS source rank is too small");
    unsigned leading = sourceType.getRank() - rhsPackedType.getRank();
    if (!llvm::all_of(sourceType.getShape().take_front(leading),
                      [](int64_t dim) { return dim == 1; }) ||
        !llvm::equal(sourceType.getShape().drop_front(leading),
                     rhsPackedType.getShape()))
      return rewriter.notifyMatchFailure(contract,
                                         "RHS source is not VNNI packed");
    for (unsigned i = 0; i < leading + 1; ++i)
      if (transpose.getPermutation()[i] != i)
        return rewriter.notifyMatchFailure(
            contract, "transpose reorders leading or K dimensions");
    if (transpose.getPermutation()[leading + 1] != leading + 2 ||
        transpose.getPermutation()[leading + 2] != leading + 1)
      return rewriter.notifyMatchFailure(
          contract, "transpose does not swap N and VNNI lanes");

    auto lhsRead = contract.getLhs().getDefiningOp<vector::TransferReadOp>();
    auto lhs = getExpandableRead(lhsRead, factor, loop);
    auto broadcast = transpose.getVector().getDefiningOp<vector::BroadcastOp>();
    Value packedSource =
        broadcast ? broadcast.getSource() : transpose.getVector();
    if (broadcast && packedSource.getType() != rhsPackedType)
      return rewriter.notifyMatchFailure(contract,
                                         "broadcast source is not VNNI packed");
    auto packedCast = packedSource.getDefiningOp<vector::ShapeCastOp>();
    auto rhsRead =
        packedCast
            ? packedCast.getSource().getDefiningOp<vector::TransferReadOp>()
            : packedSource.getDefiningOp<vector::TransferReadOp>();
    if (!lhs)
      return rewriter.notifyMatchFailure(
          contract, "LHS read cannot be expanded into VNNI layout");
    if (!isSimpleReadOutsideLoop(rhsRead, loop))
      return rewriter.notifyMatchFailure(contract,
                                         "RHS is not a simple transfer read");
    std::optional<ExpandedRead> splitRhs;
    if (packedCast) {
      auto flatType = rhsRead.getVectorType();
      if (flatType.getRank() != 2 ||
          flatType.getDimSize(0) != rhsPackedType.getDimSize(0) ||
          flatType.getDimSize(1) % factor != 0 ||
          flatType.getDimSize(1) / factor != rhsPackedType.getDimSize(1))
        return rewriter.notifyMatchFailure(
            contract, "flat RHS shape is not VNNI compatible");
      if (!(splitRhs = getExpandableRead(rhsRead, factor, loop)))
        return rewriter.notifyMatchFailure(
            contract, "RHS read cannot be expanded into VNNI layout");
    } else {
      if (rhsRead.getVectorType() != rhsPackedType)
        return rewriter.notifyMatchFailure(contract,
                                           "RHS read is not VNNI packed");
      if (getConstantIntValue(rhsRead.getIndices().back()) != 0)
        return rewriter.notifyMatchFailure(
            contract, "packed RHS has nonzero inner offset");
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
  patterns.add<CanonicalizeLoopIV, RebaseReadIndices, FoldVNNIUnpacking>(
      patterns.getContext());
}