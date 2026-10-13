// RUN: mlir-opt %s -transform-interpreter | FileCheck %s

#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (k, n)>
#acc = affine_map<(m, n, k) -> (m, n)>

// CHECK-LABEL: func.func @integer_counter(
// CHECK: %[[STEP:.+]] = arith.constant 32 : index
// CHECK: %[[LB:.+]] = arith.index_cast %{{.+}} : i32 to index
// CHECK: %[[UB:.+]] = arith.index_cast %{{.+}} : i32 to index
// CHECK: scf.for %[[K:.+]] = %[[LB]] to %[[UB]] step %[[STEP]]
// CHECK: vector.transfer_read %{{.+}}[%{{.+}}, %[[K]]]
// CHECK: vector.transfer_read %{{.+}}[%[[K]], %{{.+}}]
func.func @integer_counter(%a: memref<?x?xbf16>, %b: memref<?x?xbf16>,
                           %init: vector<16x32xf32>, %lb: i32, %ub: i32,
                           %m: index, %n: index) -> vector<16x32xf32> {
  %step = arith.constant 32 : i32
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %lb to %ub step %step iter_args(%accum = %init) -> (vector<16x32xf32>) : i32 {
    %idx = arith.index_cast %k : i32 to index
    %lhs = vector.transfer_read %a[%m, %idx], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<16x32xbf16>
    %rhs = vector.transfer_read %b[%idx, %n], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %rhs, %accum : vector<16x32xbf16>, vector<32x32xbf16> into vector<16x32xf32>
    scf.yield %dot : vector<16x32xf32>
  }
  return %result : vector<16x32xf32>
}

// CHECK-LABEL: func.func @index_round_trip(
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK-NOT: arith.index_cast
// CHECK: vector.transfer_read %{{.+}}[%{{.+}}, %[[K]]]
func.func @index_round_trip(%a: memref<?x?xbf16>, %b: memref<?x?xbf16>,
                            %init: vector<16x32xf32>, %m: index, %n: index) -> vector<16x32xf32> {
  %lb = arith.constant 0 : index
  %ub = arith.constant 64 : index
  %step = arith.constant 32 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %lb to %ub step %step iter_args(%accum = %init) -> (vector<16x32xf32>) {
    %trunc = arith.index_cast %k : index to i32
    %idx = arith.index_cast %trunc : i32 to index
    %lhs = vector.transfer_read %a[%m, %idx], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<16x32xbf16>
    %rhs = vector.transfer_read %b[%idx, %n], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %rhs, %accum : vector<16x32xbf16>, vector<32x32xbf16> into vector<16x32xf32>
    scf.yield %dot : vector<16x32xf32>
  }
  return %result : vector<16x32xf32>
}

// CHECK-LABEL: func.func @unsigned_counter(
// CHECK: scf.for unsigned %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}} {{.*}} : i32 {
// CHECK: arith.index_cast %[[K]] : i32 to index
func.func @unsigned_counter(%a: memref<?x?xbf16>, %b: memref<?x?xbf16>,
                            %init: vector<16x32xf32>, %lb: i32, %ub: i32,
                            %m: index, %n: index) -> vector<16x32xf32> {
  %step = arith.constant 32 : i32
  %pad = arith.constant 0.0 : bf16
  %result = scf.for unsigned %k = %lb to %ub step %step iter_args(%accum = %init) -> (vector<16x32xf32>) : i32 {
    %idx = arith.index_cast %k : i32 to index
    %lhs = vector.transfer_read %a[%m, %idx], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<16x32xbf16>
    %rhs = vector.transfer_read %b[%idx, %n], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %rhs, %accum : vector<16x32xbf16>, vector<32x32xbf16> into vector<16x32xf32>
    scf.yield %dot : vector<16x32xf32>
  }
  return %result : vector<16x32xf32>
}

// CHECK-LABEL: func.func @wide_counter(
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}} {{.*}} : i128 {
// CHECK: arith.index_cast %[[K]] : i128 to index
func.func @wide_counter(%a: memref<?x?xbf16>, %b: memref<?x?xbf16>,
                        %init: vector<16x32xf32>, %lb: i128, %ub: i128,
                        %m: index, %n: index) -> vector<16x32xf32> {
  %step = arith.constant 32 : i128
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %lb to %ub step %step iter_args(%accum = %init) -> (vector<16x32xf32>) : i128 {
    %idx = arith.index_cast %k : i128 to index
    %lhs = vector.transfer_read %a[%m, %idx], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<16x32xbf16>
    %rhs = vector.transfer_read %b[%idx, %n], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %rhs, %accum : vector<16x32xbf16>, vector<32x32xbf16> into vector<16x32xf32>
    scf.yield %dot : vector<16x32xf32>
  }
  return %result : vector<16x32xf32>
}

// CHECK-LABEL: func.func @integer_arithmetic(
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}} {{.*}} : i32 {
// CHECK: arith.addi %[[K]], %{{.+}} : i32
func.func @integer_arithmetic(%a: memref<?x?xbf16>, %b: memref<?x?xbf16>,
                              %init: vector<16x32xf32>, %lb: i32, %ub: i32,
                              %m: index, %n: index) -> vector<16x32xf32> {
  %step = arith.constant 32 : i32
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %lb to %ub step %step iter_args(%accum = %init) -> (vector<16x32xf32>) : i32 {
    %incremented = arith.addi %k, %step : i32
    %idx = arith.index_cast %incremented : i32 to index
    %lhs = vector.transfer_read %a[%m, %idx], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<16x32xbf16>
    %rhs = vector.transfer_read %b[%idx, %n], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %rhs, %accum : vector<16x32xbf16>, vector<32x32xbf16> into vector<16x32xf32>
    scf.yield %dot : vector<16x32xf32>
  }
  return %result : vector<16x32xf32>
}

// CHECK-LABEL: func.func @invariant_offset(
// CHECK: memref.subview %{{.+}}[0, %{{.+}}] [{{.+}}] [1, 1]
// CHECK: memref.subview %{{.+}}[%{{.+}}, 0] [{{.+}}] [1, 1]
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK: vector.transfer_read %{{.+}}[%{{.+}}, %[[K]]]
// CHECK: vector.transfer_read %{{.+}}[%[[K]], %{{.+}}]
func.func @invariant_offset(%a: memref<?x?xbf16>, %b: memref<?x?xbf16>,
                            %init: vector<16x32xf32>, %ub: index,
                            %m: index, %n: index, %off: index) -> vector<16x32xf32> {
  %lb = arith.constant 0 : index
  %step = arith.constant 32 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %lb to %ub step %step iter_args(%accum = %init) -> (vector<16x32xf32>) {
    %idx = arith.addi %k, %off : index
    %lhs = vector.transfer_read %a[%m, %idx], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<16x32xbf16>
    %rhs = vector.transfer_read %b[%idx, %n], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %rhs, %accum : vector<16x32xbf16>, vector<32x32xbf16> into vector<16x32xf32>
    scf.yield %dot : vector<16x32xf32>
  }
  return %result : vector<16x32xf32>
}

// CHECK-LABEL: func.func @shift_lower_bound(
// CHECK: memref.subview
// CHECK: memref.subview
// CHECK: scf.for %[[K:.+]] = %c0 to %{{.+}} step %c32
// CHECK: vector.transfer_read %{{.+}}[%{{.+}}, %[[K]]]
// CHECK: vector.transfer_read %{{.+}}[%[[K]], %{{.+}}]
func.func @shift_lower_bound(%a: memref<?x?xbf16>, %b: memref<?x?xbf16>,
                             %init: vector<16x32xf32>, %m: index,
                             %n: index) -> vector<16x32xf32> {
  %lb = arith.constant 32 : index
  %ub = arith.constant 96 : index
  %step = arith.constant 32 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %lb to %ub step %step iter_args(%accum = %init) -> (vector<16x32xf32>) {
    %idx = arith.subi %k, %lb : index
    %lhs = vector.transfer_read %a[%m, %idx], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<16x32xbf16>
    %rhs = vector.transfer_read %b[%idx, %n], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %rhs, %accum : vector<16x32xbf16>, vector<32x32xbf16> into vector<16x32xf32>
    scf.yield %dot : vector<16x32xf32>
  }
  return %result : vector<16x32xf32>
}

// CHECK-LABEL: func.func @unproven_shift(
// CHECK-NOT: memref.subview
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK: arith.addi %[[K]], %{{.+}} : index
func.func @unproven_shift(%a: memref<?x?xbf16>, %b: memref<?x?xbf16>,
                          %init: vector<16x32xf32>, %lb: index, %ub: index,
                          %m: index, %n: index, %off: index) -> vector<16x32xf32> {
  %step = arith.constant 32 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %lb to %ub step %step iter_args(%accum = %init) -> (vector<16x32xf32>) {
    %idx = arith.addi %k, %off : index
    %lhs = vector.transfer_read %a[%m, %idx], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<16x32xbf16>
    %rhs = vector.transfer_read %b[%idx, %n], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %rhs, %accum : vector<16x32xbf16>, vector<32x32xbf16> into vector<16x32xf32>
    scf.yield %dot : vector<16x32xf32>
  }
  return %result : vector<16x32xf32>
}

// CHECK-LABEL: func.func @non_affine_k(
// CHECK-NOT: memref.subview
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK: arith.divsi %[[K]], %{{.+}} : index
func.func @non_affine_k(%a: memref<?x?xbf16>, %b: memref<?x?xbf16>,
                        %init: vector<16x32xf32>, %ub: index,
                        %m: index, %n: index) -> vector<16x32xf32> {
  %lb = arith.constant 0 : index
  %step = arith.constant 32 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %lb to %ub step %step iter_args(%accum = %init) -> (vector<16x32xf32>) {
    %div = arith.divsi %k, %step : index
    %idx = arith.muli %div, %step : index
    %lhs = vector.transfer_read %a[%m, %idx], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<16x32xbf16>
    %rhs = vector.transfer_read %b[%idx, %n], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %rhs, %accum : vector<16x32xbf16>, vector<32x32xbf16> into vector<16x32xf32>
    scf.yield %dot : vector<16x32xf32>
  }
  return %result : vector<16x32xf32>
}

// CHECK-LABEL: func.func @two_dependent_indices(
// CHECK-NOT: memref.subview
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK: vector.transfer_read %{{.+}}[%[[K]], %[[K]]]
func.func @two_dependent_indices(%a: memref<?x?xbf16>, %b: memref<?x?xbf16>,
                                 %init: vector<16x32xf32>, %ub: index,
                                 %n: index) -> vector<16x32xf32> {
  %lb = arith.constant 0 : index
  %step = arith.constant 32 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %lb to %ub step %step iter_args(%accum = %init) -> (vector<16x32xf32>) {
    %lhs = vector.transfer_read %a[%k, %k], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<16x32xbf16>
    %rhs = vector.transfer_read %b[%k, %n], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %rhs, %accum : vector<16x32xbf16>, vector<32x32xbf16> into vector<16x32xf32>
    scf.yield %dot : vector<16x32xf32>
  }
  return %result : vector<16x32xf32>
}

// CHECK-LABEL: func.func @no_dependent_rhs(
// CHECK-NOT: memref.subview
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK: vector.transfer_read %{{.+}}[%{{.+}}, %[[K]]]
// CHECK: vector.transfer_read %{{.+}}[%{{.+}}, %{{.+}}]
func.func @no_dependent_rhs(%a: memref<?x?xbf16>, %b: memref<?x?xbf16>,
                            %init: vector<16x32xf32>, %ub: index,
                            %m: index, %n: index) -> vector<16x32xf32> {
  %lb = arith.constant 0 : index
  %step = arith.constant 32 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %lb to %ub step %step iter_args(%accum = %init) -> (vector<16x32xf32>) {
    %lhs = vector.transfer_read %a[%m, %k], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<16x32xbf16>
    %rhs = vector.transfer_read %b[%m, %n], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %rhs, %accum : vector<16x32xbf16>, vector<32x32xbf16> into vector<16x32xf32>
    scf.yield %dot : vector<16x32xf32>
  }
  return %result : vector<16x32xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.x86.prepare_vector_contract_loops
    } : !transform.any_op
    transform.yield
  }
}