//===- ShardingInterface.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MESH_INTERFACES_SHARDINGINTERFACE_H_
#define MLIR_DIALECT_MESH_INTERFACES_SHARDINGINTERFACE_H_

#include "mlir/Dialect/Mesh/IR/MeshOps.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

class Operation;
class IRMapping;
class SymbolTableCollection;

namespace mesh {

// This method retrieves the 'MeshSharding' from a given operation
// result and includes the 'annotate_for_users' information.
FailureOr<std::pair<bool, MeshSharding>> getMeshSharding(OpResult result);

// This method retrieves the 'MeshSharding' from a given operation
// operand and includes the 'annotate_for_users' information.
FailureOr<std::pair<bool, MeshSharding>> getMeshSharding(OpOperand &opOperand);

namespace detail {

FailureOr<MeshSharding>
defaultGetShardingOption(Operation *op, ArrayRef<MeshSharding> operandShardings,
                         ArrayRef<MeshSharding> resultShardings);

FailureOr<std::vector<MeshSharding>>
defaultGetShardingAnnotations(Operation *op,
                              const MeshSharding &shardingOption);

LogicalResult defaultAddShardingAnnotations(Operation *op, OpBuilder &b,
                                            const MeshSharding &shardingOption);

} // namespace detail

// Assumes full replication on all ranked tensor arguments and results.
void spmdizeFullyReplicatedOperation(Operation &op,
                                     ArrayRef<Value> spmdizedOperands,
                                     ArrayRef<MeshSharding> operandShardings,
                                     ArrayRef<MeshSharding> resultShardings,
                                     IRMapping &spmdizationMap,
                                     SymbolTableCollection &symbolTable,
                                     OpBuilder &builder);

} // namespace mesh
} // namespace mlir

/// Include the ODS generated interface header files.
#include "mlir/Dialect/Mesh/Interfaces/ShardingInterface.h.inc"

#endif // MLIR_DIALECT_MESH_INTERFACES_SHARDINGINTERFACE_H_
