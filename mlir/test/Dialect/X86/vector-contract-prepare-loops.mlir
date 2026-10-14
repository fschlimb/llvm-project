// RUN: mlir-opt %s -transform-interpreter | FileCheck %s
// RUN: mlir-opt %s -transform-interpreter=entry-point=__transform_nano | FileCheck %s --check-prefix=NANO

#lhs = affine_map<(m, n, k) -> (m, k)>
#rhs = affine_map<(m, n, k) -> (k, n)>
#acc = affine_map<(m, n, k) -> (m, n)>
#permuted_lhs = affine_map<(n, m, k) -> (m, k)>
#permuted_rhs = affine_map<(n, m, k) -> (k, n)>
#permuted_acc = affine_map<(n, m, k) -> (m, n)>

// CHECK-DAG: #[[$VNNI_LHS:[A-Za-z0-9_]+]] = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
// CHECK-DAG: #[[$VNNI_RHS:[A-Za-z0-9_]+]] = affine_map<(d0, d1, d2, d3) -> (d2, d1, d3)>
// CHECK-DAG: #[[$VNNI_ACC:[A-Za-z0-9_]+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1)>

// safe cast in the loop
// CHECK-LABEL: func.func @integer_counter(
// NANO-LABEL: func.func @integer_counter(
// NANO-NOT: vector.contract
// NANO: return
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
    %a_view = memref.cast %a : memref<?x?xbf16> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
    %lhs = vector.transfer_read %a_view[%m, %idx], %pad {in_bounds = [true, true]} : memref<?x?xbf16, strided<[?, ?], offset: ?>>, vector<16x32xbf16>
    %rhs = vector.transfer_read %b[%idx, %n], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %rhs, %accum : vector<16x32xbf16>, vector<32x32xbf16> into vector<16x32xf32>
    scf.yield %dot : vector<16x32xf32>
  }
  return %result : vector<16x32xf32>
}

// negative test: wrong order in indexing maps ("#permuted_*") -> skip
// CHECK-LABEL: func.func @permuted_maps(
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}} {{.*}} : i32 {
// CHECK: %[[IDX:.+]] = arith.index_cast %[[K]] : i32 to index
// CHECK: vector.transfer_read %{{.+}}[%{{.+}}, %[[IDX]]]
// CHECK: vector.transfer_read %{{.+}}[%[[IDX]], %{{.+}}]
func.func @permuted_maps(%a: memref<?x?xbf16>, %b: memref<?x?xbf16>,
                         %init: vector<16x32xf32>, %lb: i32, %ub: i32,
                         %m: index, %n: index) -> vector<16x32xf32> {
  %step = arith.constant 32 : i32
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %lb to %ub step %step iter_args(%accum = %init) -> (vector<16x32xf32>) : i32 {
    %idx = arith.index_cast %k : i32 to index
    %lhs = vector.transfer_read %a[%m, %idx], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<16x32xbf16>
    %rhs = vector.transfer_read %b[%idx, %n], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#permuted_lhs, #permuted_rhs, #permuted_acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %rhs, %accum : vector<16x32xbf16>, vector<32x32xbf16> into vector<16x32xf32>
    scf.yield %dot : vector<16x32xf32>
  }
  return %result : vector<16x32xf32>
}

// safe index round trip: the index cast from index to i32 and back to index is safe to eliminate
// CHECK-LABEL: func.func @index_round_trip(
// NANO-LABEL: func.func @index_round_trip(
// NANO-NOT: vector.contract
// NANO: return
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK-NOT: arith.index_cast
// CHECK: vector.transfer_read %{{.+}}[%{{.+}}, %[[K]]]
// CHECK: vector.transfer_read %{{.+}}[%[[K]], %{{.+}}]
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

// negative test: cannot prove index round trip casts are safe to eliminate
// CHECK-LABEL: func.func @unproven_round_trip(
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK: %[[NARROW:.+]] = arith.index_cast %[[K]] : index to i32
// CHECK: %[[IDX:.+]] = arith.index_cast %[[NARROW]] : i32 to index
// CHECK: vector.transfer_read %{{.+}}[%{{.+}}, %[[IDX]]]
// CHECK: vector.transfer_read %{{.+}}[%[[IDX]], %{{.+}}]
func.func @unproven_round_trip(%a: memref<?x?xbf16>, %b: memref<?x?xbf16>,
                               %init: vector<16x32xf32>, %lb: index, %ub: index,
                               %m: index, %n: index) -> vector<16x32xf32> {
  %step = arith.constant 32 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %lb to %ub step %step iter_args(%accum = %init) -> (vector<16x32xf32>) {
    %narrow = arith.index_cast %k : index to i32
    %idx = arith.index_cast %narrow : i32 to index
    %lhs = vector.transfer_read %a[%m, %idx], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<16x32xbf16>
    %rhs = vector.transfer_read %b[%idx, %n], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %rhs, %accum : vector<16x32xbf16>, vector<32x32xbf16> into vector<16x32xf32>
    scf.yield %dot : vector<16x32xf32>
  }
  return %result : vector<16x32xf32>
}

// negative test: unsigned counter casts round trip elimination not supported
// CHECK-LABEL: func.func @unsigned_counter(
// CHECK: scf.for unsigned %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}} {{.*}} : i32 {
// CHECK: %[[IDX:.+]] = arith.index_cast %[[K]] : i32 to index
// CHECK: vector.transfer_read %{{.+}}[%{{.+}}, %[[IDX]]]
// CHECK: vector.transfer_read %{{.+}}[%[[IDX]], %{{.+}}]
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

// negative test: wide counter casts round trip elimination not safe
// CHECK-LABEL: func.func @wide_counter(
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}} {{.*}} : i128 {
// CHECK: %[[IDX:.+]] = arith.index_cast %[[K]] : i128 to index
// CHECK: vector.transfer_read %{{.+}}[%{{.+}}, %[[IDX]]]
// CHECK: vector.transfer_read %{{.+}}[%[[IDX]], %{{.+}}]
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

// negative test: possible overflow due to integer arithmetic -> no elimination
// CHECK-LABEL: func.func @integer_arithmetic(
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}} {{.*}} : i32 {
// CHECK: %[[INC:.+]] = arith.addi %[[K]], %{{.+}} : i32
// CHECK: %[[IDX:.+]] = arith.index_cast %[[INC]] : i32 to index
// CHECK: vector.transfer_read %{{.+}}[%{{.+}}, %[[IDX]]]
// CHECK: vector.transfer_read %{{.+}}[%[[IDX]], %{{.+}}]
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

// "hoisting" loop-invariant offset out of loop using pre-loop subviews
// CHECK-LABEL: func.func @invariant_offset(
// NANO-LABEL: func.func @invariant_offset(
// NANO-NOT: vector.contract
// NANO: return
// CHECK: %[[A_VIEW:.+]] = memref.subview %{{.+}}[0, %{{.+}}] [{{.+}}] [1, 1]
// CHECK: %[[B_VIEW:.+]] = memref.subview %{{.+}}[%{{.+}}, 0] [{{.+}}] [1, 1]
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK: vector.transfer_read %[[A_VIEW]][%{{.+}}, %[[K]]]
// CHECK: vector.transfer_read %[[B_VIEW]][%[[K]], %{{.+}}]
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

// "hoisting" rhs-only loop-invariant offset out of loop using pre-loop subview
// CHECK-LABEL: func.func @one_offset(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x?xbf16>
// CHECK: %[[VIEW:.+]] = memref.subview
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK: vector.transfer_read %[[A]][%{{.+}}, %[[K]]]
// CHECK: vector.transfer_read %[[VIEW]][%[[K]], %{{.+}}]
func.func @one_offset(%a: memref<?x?xbf16>, %b: memref<?x?xbf16>,
                      %init: vector<16x32xf32>, %ub: index,
                      %m: index, %n: index, %off: index) -> vector<16x32xf32> {
  %lb = arith.constant 0 : index
  %step = arith.constant 32 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %lb to %ub step %step iter_args(%accum = %init) -> (vector<16x32xf32>) {
    %idx = arith.addi %k, %off : index
    %lhs = vector.transfer_read %a[%m, %k], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<16x32xbf16>
    %rhs = vector.transfer_read %b[%idx, %n], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %rhs, %accum : vector<16x32xbf16>, vector<32x32xbf16> into vector<16x32xf32>
    scf.yield %dot : vector<16x32xf32>
  }
  return %result : vector<16x32xf32>
}

// "hoisting" offset out of loop after shifting nonzero lower bound to 0
// CHECK-LABEL: func.func @shift_lower_bound(
// NANO-LABEL: func.func @shift_lower_bound(
// NANO-NOT: vector.contract
// NANO: return
// CHECK: %[[A_VIEW:.+]] = memref.subview %{{.+}}[0, %{{.+}}] [{{.+}}] [1, 1]
// CHECK: %[[B_VIEW:.+]] = memref.subview %{{.+}}[%{{.+}}, 0] [{{.+}}] [1, 1]
// CHECK: scf.for %[[K:.+]] = %c0 to %{{.+}} step %c32
// CHECK: vector.transfer_read %[[A_VIEW]][%{{.+}}, %[[K]]]
// CHECK: vector.transfer_read %[[B_VIEW]][%[[K]], %{{.+}}]
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

// negative test: cannot prove shifting dynamic lower bound to 0 is overflow-free -> skip
// CHECK-LABEL: func.func @unproven_shift(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x?xbf16>, %[[B:[^,]+]]: memref<?x?xbf16>, %{{[^,]+}}: vector<16x32xf32>, %[[LB:[^,]+]]: index
// CHECK-NOT: memref.subview
// CHECK: scf.for %[[K:.+]] = %[[LB]] to %{{.+}} step %{{.+}}
// CHECK: %[[IDX:.+]] = arith.addi %[[K]], %{{.+}} : index
// CHECK: vector.transfer_read %[[A]][%{{.+}}, %[[IDX]]]
// CHECK: vector.transfer_read %[[B]][%[[IDX]], %{{.+}}]
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

// negative test: non-strided lhs memref layout cannot be subviewed -> skip
// CHECK-LABEL: func.func @non_strided_layout(
// CHECK-SAME: %[[A:[^,]+]]: memref<64x64xbf16, #{{.+}}>, %[[B:[^,]+]]: memref<64x64xbf16>
// CHECK-NOT: memref.subview
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK: %[[IDX:.+]] = arith.addi %[[K]], %{{.+}} : index
// CHECK: vector.transfer_read %[[A]][%{{.+}}, %[[IDX]]]
// CHECK: vector.transfer_read %[[B]][%[[IDX]], %{{.+}}]
func.func @non_strided_layout(%a: memref<64x64xbf16, affine_map<(d0, d1) -> (d1, d0)>>,
                              %b: memref<64x64xbf16>, %init: vector<16x32xf32>,
                              %ub: index, %m: index, %n: index,
                              %off: index) -> vector<16x32xf32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 32 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<16x32xf32>) {
    %idx = arith.addi %k, %off : index
    %lhs = vector.transfer_read %a[%m, %idx], %pad {in_bounds = [true, true]} : memref<64x64xbf16, affine_map<(d0, d1) -> (d1, d0)>>, vector<16x32xbf16>
    %rhs = vector.transfer_read %b[%idx, %n], %pad {in_bounds = [true, true]} : memref<64x64xbf16>, vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %rhs, %accum : vector<16x32xbf16>, vector<32x32xbf16> into vector<16x32xf32>
    scf.yield %dot : vector<16x32xf32>
  }
  return %result : vector<16x32xf32>
}

// fold exact round-downs (k / 32) * 32 and 16 * (k floordiv 16) to k
// CHECK-LABEL: func.func @round_down_k(
// NANO-LABEL: func.func @round_down_k(
// NANO-NOT: vector.contract
// NANO: return
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK-NOT: arith.muli
// CHECK: vector.transfer_read %{{.+}}[%{{.+}}, %[[K]]]
// CHECK: vector.transfer_read %{{.+}}[%[[K]], %{{.+}}]
func.func @round_down_k(%a: memref<?x?xbf16>, %b: memref<?x?xbf16>,
                        %init: vector<16x32xf32>, %ub: index,
                        %m: index, %n: index) -> vector<16x32xf32> {
  %lb = arith.constant 0 : index
  %step = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %lb to %ub step %step iter_args(%accum = %init) -> (vector<16x32xf32>) {
    %div = arith.divsi %k, %step : index
    %lhs_idx = arith.muli %div, %step : index
    %floor = arith.floordivsi %k, %c16 : index
    %rhs_idx = arith.muli %c16, %floor : index
    %lhs = vector.transfer_read %a[%m, %lhs_idx], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<16x32xbf16>
    %rhs = vector.transfer_read %b[%rhs_idx, %n], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %rhs, %accum : vector<16x32xbf16>, vector<32x32xbf16> into vector<16x32xf32>
    scf.yield %dot : vector<16x32xf32>
  }
  return %result : vector<16x32xf32>
}

// fold exact round-down (k ceildiv 16) * 16 to k with a tiled lower bound
// CHECK-LABEL: func.func @round_down_tiled_k(
// CHECK: scf.for %[[KK:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK: scf.for %[[K:.+]] = %[[KK]] to %{{.+}} step %{{.+}}
// CHECK-NOT: arith.muli
// CHECK: vector.transfer_read %{{.+}}[%{{.+}}, %[[K]]]
// CHECK: vector.transfer_read %{{.+}}[%[[K]], %{{.+}}]
func.func @round_down_tiled_k(%a: memref<?x?xbf16>, %b: memref<?x?xbf16>,
                              %init: vector<16x32xf32>, %ub: index,
                              %m: index, %n: index) -> vector<16x32xf32> {
  %zero = arith.constant 0 : index
  %tile = arith.constant 128 : index
  %step = arith.constant 32 : index
  %c16 = arith.constant 16 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %kk = %zero to %ub step %tile iter_args(%outer = %init) -> (vector<16x32xf32>) {
    %end = arith.addi %kk, %tile : index
    %inner = scf.for %k = %kk to %end step %step iter_args(%accum = %outer) -> (vector<16x32xf32>) {
      %ceil = arith.ceildivsi %k, %c16 : index
      %idx = arith.muli %ceil, %c16 : index
      %lhs = vector.transfer_read %a[%m, %idx], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<16x32xbf16>
      %rhs = vector.transfer_read %b[%idx, %n], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x32xbf16>
      %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %rhs, %accum : vector<16x32xbf16>, vector<32x32xbf16> into vector<16x32xf32>
      scf.yield %dot : vector<16x32xf32>
    }
    scf.yield %inner : vector<16x32xf32>
  }
  return %result : vector<16x32xf32>
}

// negative test: (k / 64) * 64 is not k for step 32 -> skip
// CHECK-LABEL: func.func @non_affine_k(
// CHECK-NOT: memref.subview
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK: %[[DIV:.+]] = arith.divsi %[[K]], %{{.+}} : index
// CHECK: %[[IDX:.+]] = arith.muli %[[DIV]], %{{.+}} : index
// CHECK: vector.transfer_read %{{.+}}[%{{.+}}, %[[IDX]]]
// CHECK: vector.transfer_read %{{.+}}[%[[IDX]], %{{.+}}]
func.func @non_affine_k(%a: memref<?x?xbf16>, %b: memref<?x?xbf16>,
                        %init: vector<16x32xf32>, %ub: index,
                        %m: index, %n: index) -> vector<16x32xf32> {
  %lb = arith.constant 0 : index
  %step = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %lb to %ub step %step iter_args(%accum = %init) -> (vector<16x32xf32>) {
    %div = arith.divsi %k, %c64 : index
    %idx = arith.muli %div, %c64 : index
    %lhs = vector.transfer_read %a[%m, %idx], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<16x32xbf16>
    %rhs = vector.transfer_read %b[%idx, %n], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %rhs, %accum : vector<16x32xbf16>, vector<32x32xbf16> into vector<16x32xf32>
    scf.yield %dot : vector<16x32xf32>
  }
  return %result : vector<16x32xf32>
}

// negative test: floordivsi of possibly wrapping k * 2 is not k -> skip
// CHECK-LABEL: func.func @wrapping_index(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x?xbf16>, %[[B:[^,]+]]: memref<?x?xbf16>
// CHECK-NOT: memref.subview
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK: %[[TWICE:.+]] = arith.muli %[[K]], %{{.+}} : index
// CHECK: %[[HALF:.+]] = arith.floordivsi %[[TWICE]], %{{.+}} : index
// CHECK: %[[IDX:.+]] = arith.addi %[[HALF]], %{{.+}} : index
// CHECK: vector.transfer_read %[[A]][%{{.+}}, %[[IDX]]]
// CHECK: vector.transfer_read %[[B]][%[[IDX]], %{{.+}}]
func.func @wrapping_index(%a: memref<?x?xbf16>, %b: memref<?x?xbf16>,
                          %init: vector<16x32xf32>, %ub: index,
                          %m: index, %n: index, %off: index) -> vector<16x32xf32> {
  %lb = arith.constant 0 : index
  %step = arith.constant 32 : index
  %c2 = arith.constant 2 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %lb to %ub step %step iter_args(%accum = %init) -> (vector<16x32xf32>) {
    %twice = arith.muli %k, %c2 : index
    %half = arith.floordivsi %twice, %c2 : index
    %idx = arith.addi %half, %off : index
    %lhs = vector.transfer_read %a[%m, %idx], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<16x32xbf16>
    %rhs = vector.transfer_read %b[%idx, %n], %pad {in_bounds = [true, true]} : memref<?x?xbf16>, vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %rhs, %accum : vector<16x32xbf16>, vector<32x32xbf16> into vector<16x32xf32>
    scf.yield %dot : vector<16x32xf32>
  }
  return %result : vector<16x32xf32>
}

// negative test: lhs read has two IV-dependent indices -> skip
// CHECK-LABEL: func.func @two_dependent_indices(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x?xbf16>, %[[B:[^,]+]]: memref<?x?xbf16>
// CHECK-NOT: memref.subview
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK: vector.transfer_read %[[A]][%[[K]], %[[K]]]
// CHECK: vector.transfer_read %[[B]][%[[K]], %{{.+}}]
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

// negative test: rhs read does not depend on IV -> skip
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

// fold VNNI unpacking of packed bf16 rhs read into contraction, keeping fastmath
// CHECK-LABEL: func.func @direct_vnni_bf16(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x32x32xbf16>, %[[B:[^,]+]]: memref<?x16x32x2xbf16>
// NANO-LABEL: func.func @direct_vnni_bf16(
// NANO-NOT: vector.contract
// NANO: return
// CHECK: %[[A_PACKED:.+]] = memref.expand_shape %[[A]] {{.*}} into memref<?x32x16x2xbf16>
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A_PACKED]][%[[K]], {{.*}} vector<32x16x2xbf16>
// CHECK: %[[RHS:.+]] = vector.transfer_read %[[B]][%[[K]], {{.*}} vector<16x32x2xbf16>
// CHECK: vector.contract {fastmath = #arith.fastmath<reassoc>, indexing_maps = [#[[$VNNI_LHS]], #[[$VNNI_RHS]], #[[$VNNI_ACC]]], iterator_types = ["parallel", "parallel", "reduction", "reduction"], kind = #vector.kind<add>} %[[LHS]], %[[RHS]], %{{.+}} : vector<32x16x2xbf16>, vector<16x32x2xbf16> into vector<32x32xf32>
func.func @direct_vnni_bf16(%a: memref<?x32x32xbf16>,
                            %b: memref<?x16x32x2xbf16>,
                            %init: vector<32x32xf32>, %ub: index,
                            %m: index) -> vector<32x32xf32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 16 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xf32>) {
    %lhs = vector.transfer_read %a[%k, %m, %zero], %pad {in_bounds = [true, true]} : memref<?x32x32xbf16>, vector<32x32xbf16>
    %rhs = vector.transfer_read %b[%k, %zero, %zero, %zero], %pad {in_bounds = [true, true, true]} : memref<?x16x32x2xbf16>, vector<16x32x2xbf16>
    %transpose = vector.transpose %rhs, [0, 2, 1] : vector<16x32x2xbf16> to vector<16x2x32xbf16>
    %flat = vector.shape_cast %transpose : vector<16x2x32xbf16> to vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>, fastmath = #arith.fastmath<reassoc>} %lhs, %flat, %accum : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
    scf.yield %dot : vector<32x32xf32>
  }
  return %result : vector<32x32xf32>
}

// fold VNNI unpacking of packed bf16 rhs read with leading unit dims
// CHECK-LABEL: func.func @leading_unit_vnni(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x32x32xbf16>, %[[B:[^,]+]]: memref<?x1x1x16x32x2xbf16>
// CHECK: %[[A_PACKED:.+]] = memref.expand_shape %[[A]] {{.*}} into memref<?x32x16x2xbf16>
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A_PACKED]][%[[K]], {{.*}} vector<32x16x2xbf16>
// CHECK: %[[RHS:.+]] = vector.transfer_read %[[B]][%[[K]], {{.*}} vector<16x32x2xbf16>
// CHECK: vector.contract {indexing_maps = [#[[$VNNI_LHS]], #[[$VNNI_RHS]], #[[$VNNI_ACC]]], iterator_types = ["parallel", "parallel", "reduction", "reduction"], kind = #vector.kind<add>} %[[LHS]], %[[RHS]], %{{.+}} : vector<32x16x2xbf16>, vector<16x32x2xbf16> into vector<32x32xf32>
// NANO-LABEL: func.func @leading_unit_vnni(
// NANO-NOT: vector.contract
// NANO: return
func.func @leading_unit_vnni(%a: memref<?x32x32xbf16>,
                             %b: memref<?x1x1x16x32x2xbf16>,
                             %init: vector<32x32xf32>, %ub: index) -> vector<32x32xf32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 16 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xf32>) {
    %lhs = vector.transfer_read %a[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x32x32xbf16>, vector<32x32xbf16>
    %rhs = vector.transfer_read %b[%k, %zero, %zero, %zero, %zero, %zero], %pad {in_bounds = [true, true, true, true, true]} : memref<?x1x1x16x32x2xbf16>, vector<1x1x16x32x2xbf16>
    %transpose = vector.transpose %rhs, [0, 1, 2, 4, 3] : vector<1x1x16x32x2xbf16> to vector<1x1x16x2x32xbf16>
    %flat = vector.shape_cast %transpose : vector<1x1x16x2x32xbf16> to vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
    scf.yield %dot : vector<32x32xf32>
  }
  return %result : vector<32x32xf32>
}

// fold VNNI unpacking of flat bf16 rhs read split into lanes by shape_cast
// CHECK-LABEL: func.func @split_vnni_bf16(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x32x32xbf16>, %[[B:[^,]+]]: memref<?x16x64xbf16>
// NANO-LABEL: func.func @split_vnni_bf16(
// NANO-NOT: vector.contract
// NANO: return
// CHECK: %[[A_PACKED:.+]] = memref.expand_shape %[[A]] {{.*}} into memref<?x32x16x2xbf16>
// CHECK: %[[B_PACKED:.+]] = memref.expand_shape %[[B]] {{.*}} into memref<?x16x32x2xbf16>
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A_PACKED]][%[[K]], {{.*}} vector<32x16x2xbf16>
// CHECK: %[[RHS:.+]] = vector.transfer_read %[[B_PACKED]][%[[K]], {{.*}} vector<16x32x2xbf16>
// CHECK: vector.contract {indexing_maps = [#[[$VNNI_LHS]], #[[$VNNI_RHS]], #[[$VNNI_ACC]]], iterator_types = ["parallel", "parallel", "reduction", "reduction"], kind = #vector.kind<add>} %[[LHS]], %[[RHS]], %{{.+}} : vector<32x16x2xbf16>, vector<16x32x2xbf16> into vector<32x32xf32>
func.func @split_vnni_bf16(%a: memref<?x32x32xbf16>,
                           %b: memref<?x16x64xbf16>,
                           %init: vector<32x32xf32>, %ub: index,
                           %m: index) -> vector<32x32xf32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 16 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xf32>) {
    %lhs = vector.transfer_read %a[%k, %m, %zero], %pad {in_bounds = [true, true]} : memref<?x32x32xbf16>, vector<32x32xbf16>
    %rhs = vector.transfer_read %b[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x16x64xbf16>, vector<16x64xbf16>
    %packed = vector.shape_cast %rhs : vector<16x64xbf16> to vector<16x32x2xbf16>
    %transpose = vector.transpose %packed, [0, 2, 1] : vector<16x32x2xbf16> to vector<16x2x32xbf16>
    %flat = vector.shape_cast %transpose : vector<16x2x32xbf16> to vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
    scf.yield %dot : vector<32x32xf32>
  }
  return %result : vector<32x32xf32>
}

// fold VNNI unpacking of packed i8 rhs read into contraction
// CHECK-LABEL: func.func @direct_vnni_int8(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x32x64xi8>, %[[B:[^,]+]]: memref<?x16x32x4xi8>
// NANO-LABEL: func.func @direct_vnni_int8(
// NANO-NOT: vector.contract
// NANO: return
// CHECK: %[[A_PACKED:.+]] = memref.expand_shape %[[A]] {{.*}} into memref<?x32x16x4xi8>
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A_PACKED]][%[[K]], {{.*}} vector<32x16x4xi8>
// CHECK: %[[RHS:.+]] = vector.transfer_read %[[B]][%[[K]], {{.*}} vector<16x32x4xi8>
// CHECK: vector.contract {indexing_maps = [#[[$VNNI_LHS]], #[[$VNNI_RHS]], #[[$VNNI_ACC]]], iterator_types = ["parallel", "parallel", "reduction", "reduction"], kind = #vector.kind<add>} %[[LHS]], %[[RHS]], %{{.+}} : vector<32x16x4xi8>, vector<16x32x4xi8> into vector<32x32xi32>
func.func @direct_vnni_int8(%a: memref<?x32x64xi8>,
                            %b: memref<?x16x32x4xi8>,
                            %init: vector<32x32xi32>, %ub: index,
                            %m: index) -> vector<32x32xi32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 16 : index
  %pad = arith.constant 0 : i8
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xi32>) {
    %lhs = vector.transfer_read %a[%k, %m, %zero], %pad {in_bounds = [true, true]} : memref<?x32x64xi8>, vector<32x64xi8>
    %rhs = vector.transfer_read %b[%k, %zero, %zero, %zero], %pad {in_bounds = [true, true, true]} : memref<?x16x32x4xi8>, vector<16x32x4xi8>
    %transpose = vector.transpose %rhs, [0, 2, 1] : vector<16x32x4xi8> to vector<16x4x32xi8>
    %flat = vector.shape_cast %transpose : vector<16x4x32xi8> to vector<64x32xi8>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x64xi8>, vector<64x32xi8> into vector<32x32xi32>
    scf.yield %dot : vector<32x32xi32>
  }
  return %result : vector<32x32xi32>
}

// fold VNNI unpacking of flat i8 rhs read split into lanes by shape_cast
// CHECK-LABEL: func.func @split_vnni_int8(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x32x64xi8>, %[[B:[^,]+]]: memref<?x16x128xi8>
// NANO-LABEL: func.func @split_vnni_int8(
// NANO-NOT: vector.contract
// NANO: return
// CHECK: %[[A_PACKED:.+]] = memref.expand_shape %[[A]] {{.*}} into memref<?x32x16x4xi8>
// CHECK: %[[B_PACKED:.+]] = memref.expand_shape %[[B]] {{.*}} into memref<?x16x32x4xi8>
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A_PACKED]][%[[K]], {{.*}} vector<32x16x4xi8>
// CHECK: %[[RHS:.+]] = vector.transfer_read %[[B_PACKED]][%[[K]], {{.*}} vector<16x32x4xi8>
// CHECK: vector.contract {indexing_maps = [#[[$VNNI_LHS]], #[[$VNNI_RHS]], #[[$VNNI_ACC]]], iterator_types = ["parallel", "parallel", "reduction", "reduction"], kind = #vector.kind<add>} %[[LHS]], %[[RHS]], %{{.+}} : vector<32x16x4xi8>, vector<16x32x4xi8> into vector<32x32xi32>
func.func @split_vnni_int8(%a: memref<?x32x64xi8>,
                           %b: memref<?x16x128xi8>,
                           %init: vector<32x32xi32>, %ub: index,
                           %m: index) -> vector<32x32xi32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 16 : index
  %pad = arith.constant 0 : i8
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xi32>) {
    %lhs = vector.transfer_read %a[%k, %m, %zero], %pad {in_bounds = [true, true]} : memref<?x32x64xi8>, vector<32x64xi8>
    %rhs = vector.transfer_read %b[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x16x128xi8>, vector<16x128xi8>
    %packed = vector.shape_cast %rhs : vector<16x128xi8> to vector<16x32x4xi8>
    %transpose = vector.transpose %packed, [0, 2, 1] : vector<16x32x4xi8> to vector<16x4x32xi8>
    %flat = vector.shape_cast %transpose : vector<16x4x32xi8> to vector<64x32xi8>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x64xi8>, vector<64x32xi8> into vector<32x32xi32>
    scf.yield %dot : vector<32x32xi32>
  }
  return %result : vector<32x32xi32>
}

// cast away leading unit dims of reads so that the contraction can be unrolled
// CHECK-LABEL: func.func @unit_dimension_reads(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x?x32x32xbf16>, %[[B:[^,]+]]: memref<?x?x32x32xbf16>
// CHECK: scf.for %[[K:.+]] = %{{.+}} to %{{.+}} step %{{.+}}
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A]][%[[K]], {{.*}} vector<32x32xbf16>
// CHECK: %[[RHS:.+]] = vector.transfer_read %[[B]][%[[K]], {{.*}} vector<32x32xbf16>
// CHECK: vector.contract {{.*}} %[[LHS]], %[[RHS]], %{{.+}} : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
// NANO-LABEL: func.func @unit_dimension_reads(
// NANO-NOT: vector.contract
// NANO: return
func.func @unit_dimension_reads(%a: memref<?x?x32x32xbf16>,
                                %b: memref<?x?x32x32xbf16>,
                                %init: vector<32x32xf32>, %ub: index,
                                %m: index, %n: index) -> vector<32x32xf32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 32 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xf32>) {
    %lhs = vector.transfer_read %a[%k, %m, %zero, %zero], %pad {in_bounds = [true, true, true, true]} : memref<?x?x32x32xbf16>, vector<1x1x32x32xbf16>
    %rhs = vector.transfer_read %b[%k, %n, %zero, %zero], %pad {in_bounds = [true, true, true, true]} : memref<?x?x32x32xbf16>, vector<1x1x32x32xbf16>
    %left = vector.shape_cast %lhs : vector<1x1x32x32xbf16> to vector<32x32xbf16>
    %right = vector.shape_cast %rhs : vector<1x1x32x32xbf16> to vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %left, %right, %accum : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
    scf.yield %dot : vector<32x32xf32>
  }
  return %result : vector<32x32xf32>
}

// negative test: lhs innermost dim is dynamic and might not be a multiple of VNNI factor -> skip
// CHECK-LABEL: func.func @dynamic_inner_k(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x32x?xbf16>
// CHECK-NOT: memref.expand_shape
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A]][{{.*}} vector<32x32xbf16>
// CHECK: %[[T:.+]] = vector.transpose
// CHECK: %[[FLAT:.+]] = vector.shape_cast %[[T]] : vector<16x2x32xbf16> to vector<32x32xbf16>
// CHECK: vector.contract {{.*}} %[[LHS]], %[[FLAT]], %{{.+}} : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
func.func @dynamic_inner_k(%a: memref<?x32x?xbf16>,
                           %b: memref<?x16x32x2xbf16>,
                           %init: vector<32x32xf32>, %ub: index,
                           %m: index) -> vector<32x32xf32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 16 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xf32>) {
    %lhs = vector.transfer_read %a[%k, %m, %zero], %pad {in_bounds = [true, true]} : memref<?x32x?xbf16>, vector<32x32xbf16>
    %rhs = vector.transfer_read %b[%k, %zero, %zero, %zero], %pad {in_bounds = [true, true, true]} : memref<?x16x32x2xbf16>, vector<16x32x2xbf16>
    %transpose = vector.transpose %rhs, [0, 2, 1] : vector<16x32x2xbf16> to vector<16x2x32xbf16>
    %flat = vector.shape_cast %transpose : vector<16x2x32xbf16> to vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
    scf.yield %dot : vector<32x32xf32>
  }
  return %result : vector<32x32xf32>
}

// negative test: lhs innermost read offset 1 is not a multiple of VNNI factor 2 -> skip
// CHECK-LABEL: func.func @misaligned_inner_k(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x32x64xbf16>
// CHECK-NOT: memref.expand_shape
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A]][{{.*}} vector<32x32xbf16>
// CHECK: %[[T:.+]] = vector.transpose
// CHECK: %[[FLAT:.+]] = vector.shape_cast %[[T]] : vector<16x2x32xbf16> to vector<32x32xbf16>
// CHECK: vector.contract {{.*}} %[[LHS]], %[[FLAT]], %{{.+}} : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
func.func @misaligned_inner_k(%a: memref<?x32x64xbf16>,
                              %b: memref<?x16x32x2xbf16>,
                              %init: vector<32x32xf32>, %ub: index) -> vector<32x32xf32> {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %step = arith.constant 16 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xf32>) {
    %lhs = vector.transfer_read %a[%k, %zero, %one], %pad {in_bounds = [true, true]} : memref<?x32x64xbf16>, vector<32x32xbf16>
    %rhs = vector.transfer_read %b[%k, %zero, %zero, %zero], %pad {in_bounds = [true, true, true]} : memref<?x16x32x2xbf16>, vector<16x32x2xbf16>
    %transpose = vector.transpose %rhs, [0, 2, 1] : vector<16x32x2xbf16> to vector<16x2x32xbf16>
    %flat = vector.shape_cast %transpose : vector<16x2x32xbf16> to vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
    scf.yield %dot : vector<32x32xf32>
  }
  return %result : vector<32x32xf32>
}

// negative test: lhs innermost dim 33 is not divisible by VNNI factor 2 -> skip
// CHECK-LABEL: func.func @nondivisible_inner_k(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x32x33xbf16>
// CHECK-NOT: memref.expand_shape
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A]][{{.*}} vector<32x32xbf16>
// CHECK: %[[T:.+]] = vector.transpose
// CHECK: %[[FLAT:.+]] = vector.shape_cast %[[T]] : vector<16x2x32xbf16> to vector<32x32xbf16>
// CHECK: vector.contract {{.*}} %[[LHS]], %[[FLAT]], %{{.+}} : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
func.func @nondivisible_inner_k(%a: memref<?x32x33xbf16>,
                                %b: memref<?x16x32x2xbf16>,
                                %init: vector<32x32xf32>, %ub: index) -> vector<32x32xf32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 16 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xf32>) {
    %lhs = vector.transfer_read %a[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x32x33xbf16>, vector<32x32xbf16>
    %rhs = vector.transfer_read %b[%k, %zero, %zero, %zero], %pad {in_bounds = [true, true, true]} : memref<?x16x32x2xbf16>, vector<16x32x2xbf16>
    %transpose = vector.transpose %rhs, [0, 2, 1] : vector<16x32x2xbf16> to vector<16x2x32xbf16>
    %flat = vector.shape_cast %transpose : vector<16x2x32xbf16> to vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
    scf.yield %dot : vector<32x32xf32>
  }
  return %result : vector<32x32xf32>
}

// negative test: rhs VNNI lanes are reordered before unpacking -> skip
// CHECK-LABEL: func.func @interleaved_lanes(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x32x64xi8>
// CHECK-NOT: memref.expand_shape
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A]][{{.*}} vector<32x64xi8>
// CHECK: %[[EVEN:[^,]+]], %[[ODD:.+]] = vector.deinterleave
// CHECK: %[[LANES:.+]] = vector.interleave %[[ODD]], %[[EVEN]]
// CHECK: %[[T:.+]] = vector.transpose %[[LANES]], [0, 2, 1]
// CHECK: %[[FLAT:.+]] = vector.shape_cast %[[T]] : vector<16x4x32xi8> to vector<64x32xi8>
// CHECK: vector.contract {{.*}} %[[LHS]], %[[FLAT]], %{{.+}} : vector<32x64xi8>, vector<64x32xi8> into vector<32x32xi32>
func.func @interleaved_lanes(%a: memref<?x32x64xi8>,
                             %b: memref<?x16x32x4xi8>,
                             %init: vector<32x32xi32>, %ub: index) -> vector<32x32xi32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 16 : index
  %pad = arith.constant 0 : i8
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xi32>) {
    %lhs = vector.transfer_read %a[%k, %zero, %zero], %pad {in_bounds = [true, true]} : memref<?x32x64xi8>, vector<32x64xi8>
    %rhs = vector.transfer_read %b[%k, %zero, %zero, %zero], %pad {in_bounds = [true, true, true]} : memref<?x16x32x4xi8>, vector<16x32x4xi8>
    %even, %odd = vector.deinterleave %rhs : vector<16x32x4xi8> -> vector<16x32x2xi8>
    %reordered = vector.interleave %odd, %even : vector<16x32x2xi8> -> vector<16x32x4xi8>
    %transpose = vector.transpose %reordered, [0, 2, 1] : vector<16x32x4xi8> to vector<16x4x32xi8>
    %flat = vector.shape_cast %transpose : vector<16x4x32xi8> to vector<64x32xi8>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x64xi8>, vector<64x32xi8> into vector<32x32xi32>
    scf.yield %dot : vector<32x32xi32>
  }
  return %result : vector<32x32xi32>
}

// negative test: scalable lhs cannot be packed into a fixed VNNI shape -> skip
// CHECK-LABEL: func.func @scalable_vnni(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x?x32xbf16>
// CHECK-NOT: memref.expand_shape
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A]][{{.*}} vector<[32]x32xbf16>
// CHECK: %[[T:.+]] = vector.transpose
// CHECK: %[[FLAT:.+]] = vector.shape_cast %[[T]] : vector<16x2x32xbf16> to vector<32x32xbf16>
// CHECK: vector.contract {{.*}} %[[LHS]], %[[FLAT]], %{{.+}} : vector<[32]x32xbf16>, vector<32x32xbf16> into vector<[32]x32xf32>
func.func @scalable_vnni(%a: memref<?x?x32xbf16>,
                         %b: memref<?x16x32x2xbf16>,
                         %init: vector<[32]x32xf32>, %ub: index,
                         %m: index) -> vector<[32]x32xf32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 16 : index
  %pad = arith.constant 0.0 : bf16
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<[32]x32xf32>) {
    %lhs = vector.transfer_read %a[%k, %m, %zero], %pad {in_bounds = [true, true]} : memref<?x?x32xbf16>, vector<[32]x32xbf16>
    %rhs = vector.transfer_read %b[%k, %zero, %zero, %zero], %pad {in_bounds = [true, true, true]} : memref<?x16x32x2xbf16>, vector<16x32x2xbf16>
    %transpose = vector.transpose %rhs, [0, 2, 1] : vector<16x32x2xbf16> to vector<16x2x32xbf16>
    %flat = vector.shape_cast %transpose : vector<16x2x32xbf16> to vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<[32]x32xbf16>, vector<32x32xbf16> into vector<[32]x32xf32>
    scf.yield %dot : vector<[32]x32xf32>
  }
  return %result : vector<[32]x32xf32>
}

// negative test: lhs memref with vector elements cannot be expanded into VNNI layout -> skip
// CHECK-LABEL: func.func @vector_element_lhs(
// CHECK-SAME: %[[A:[^,]+]]: memref<?x32xvector<32xbf16>>
// CHECK-NOT: memref.expand_shape
// CHECK: %[[LHS:.+]] = vector.transfer_read %[[A]][{{.*}} vector<32x32xbf16>
// CHECK: %[[T:.+]] = vector.transpose
// CHECK: %[[FLAT:.+]] = vector.shape_cast %[[T]] : vector<16x2x32xbf16> to vector<32x32xbf16>
// CHECK: vector.contract {{.*}} %[[LHS]], %[[FLAT]], %{{.+}} : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
func.func @vector_element_lhs(%a: memref<?x32xvector<32xbf16>>,
                              %b: memref<?x16x32x2xbf16>,
                              %init: vector<32x32xf32>, %ub: index) -> vector<32x32xf32> {
  %zero = arith.constant 0 : index
  %step = arith.constant 16 : index
  %pad = arith.constant 0.0 : bf16
  %vpad = arith.constant dense<0.0> : vector<32xbf16>
  %result = scf.for %k = %zero to %ub step %step iter_args(%accum = %init) -> (vector<32x32xf32>) {
    %lhs = vector.transfer_read %a[%k, %zero], %vpad {in_bounds = [true]} : memref<?x32xvector<32xbf16>>, vector<32x32xbf16>
    %rhs = vector.transfer_read %b[%k, %zero, %zero, %zero], %pad {in_bounds = [true, true, true]} : memref<?x16x32x2xbf16>, vector<16x32x2xbf16>
    %transpose = vector.transpose %rhs, [0, 2, 1] : vector<16x32x2xbf16> to vector<16x2x32xbf16>
    %flat = vector.shape_cast %transpose : vector<16x2x32xbf16> to vector<32x32xbf16>
    %dot = vector.contract {indexing_maps = [#lhs, #rhs, #acc], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %lhs, %flat, %accum : vector<32x32xbf16>, vector<32x32xbf16> into vector<32x32xf32>
    scf.yield %dot : vector<32x32xf32>
  }
  return %result : vector<32x32xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %loops = transform.structured.match ops{["scf.for"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %loops : !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
      transform.apply_patterns.x86.prepare_vector_contract_loops
    } : !transform.any_op
    transform.apply_dce to %funcs : !transform.any_op
    transform.apply_cse to %funcs : !transform.any_op
    transform.yield
  }

  transform.named_sequence @__transform_nano(%root: !transform.any_op {transform.readonly}) {
    %funcs = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %loops = transform.structured.match ops{["scf.for"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %loops : !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
      transform.apply_patterns.x86.prepare_vector_contract_loops
    } : !transform.any_op
    transform.apply_dce to %funcs : !transform.any_op
    transform.apply_cse to %funcs : !transform.any_op
    transform.apply_patterns to %funcs {
      transform.apply_patterns.x86.vector_contract_multi_level_unroll enable_amx
      transform.apply_patterns.x86.vector_contract_to_amx_dot_product
    } : !transform.any_op
    transform.yield
  }
}