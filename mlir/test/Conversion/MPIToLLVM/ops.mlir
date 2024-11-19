// RUN: mlir-opt -convert-to-llvm %s | FileCheck %s

module {
  // CHECK: llvm.func @MPI_Finalize() -> i32
  // CHECK: llvm.func @MPI_recv(!llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32
  // CHECK: llvm.func @MPI_send(!llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: llvm.func @MPI_Comm_rank({{.*}}, !llvm.ptr) -> i32
  // COMM: llvm.mlir.global external @MPI_COMM_WORLD() {addr_space = 0 : i32} : !llvm.struct<"MPI_ABI_Comm", opaque>
  // CHECK: llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32

  func.func @mpi_test(%arg0: memref<100xf32>) {
    // CHECK: [[v7:%.*]] = builtin.unrealized_conversion_cast %{{.+}} : memref<100xf32> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // CHECK-NEXT: [[v8:%.*]] = llvm.mlir.zero : !llvm.ptr
    // CHECK-NEXT: [[v9:%.*]] = llvm.call @MPI_Init([[v8]], [[v8]]) : (!llvm.ptr, !llvm.ptr) -> i32
    // CHECK-NEXT: [[v10:%.*]] = builtin.unrealized_conversion_cast [[v9]] : i32 to !mpi.retval
    %0 = mpi.init : !mpi.retval

    // CHECK: [[v12:%.*]] = llvm.mlir.constant(1 : i32) : i32
    // CHECK-NEXT: [[v13:%.*]] = llvm.alloca [[v12]] x i32 : (i32) -> !llvm.ptr
    // CHECK-NEXT: [[v14:%.*]] = llvm.call @MPI_Comm_rank(%{{.+}}, [[v13]]) : (i32, !llvm.ptr) -> i32
    // CHECK-NEXT: [[v15:%.*]] = builtin.unrealized_conversion_cast [[v14]] : i32 to !mpi.retval
    // CHECK-NEXT: [[v16:%.*]] = llvm.load [[v13]] : !llvm.ptr -> i32
    %retval, %rank = mpi.comm_rank : !mpi.retval, i32

    // CHECK: [[v17:%.*]] = llvm.extractvalue [[v7]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK-NEXT: [[v18:%.*]] = llvm.extractvalue [[v7]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK-NEXT: [[v19:%.*]] = llvm.getelementptr [[v17]][[[v18]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK-NEXT: [[v20:%.*]] = llvm.mlir.constant
    // CHECK-NEXT: [[v21:%.*]] = llvm.mlir.constant
    // CHECK-NEXT: [[v22:%.*]] = llvm.call @MPI_send([[v19]], [[v20]], [[v16]], [[v16]], [[v21]]) : (!llvm.ptr, i32, i32, i32, i32) -> i32
    mpi.send(%arg0, %rank, %rank) : memref<100xf32>, i32, i32

    // CHECK-NEXT: [[v23:%.*]] = llvm.extractvalue [[v7]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK-NEXT: [[v24:%.*]] = llvm.extractvalue [[v7]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK-NEXT: [[v25:%.*]] = llvm.getelementptr [[v23]][[[v24]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK-NEXT: [[v26:%.*]] = llvm.mlir.constant
    // CHECK-NEXT: [[v27:%.*]] = llvm.mlir.constant
    // CHECK-NEXT: [[v28:%.*]] = llvm.call @MPI_send([[v25]], [[v26]], [[v16]], [[v16]], [[v27]]) : (!llvm.ptr, i32, i32, i32, i32) -> i32
    %1 = mpi.send(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval

    // CHECK: [[v29:%.*]] = llvm.extractvalue [[v7]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK-NEXT: [[v30:%.*]] = llvm.extractvalue [[v7]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK-NEXT: [[v31:%.*]] = llvm.getelementptr [[v29]][[[v30]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK-NEXT: [[v32:%.*]] = llvm.mlir.constant
    // CHECK-NEXT: [[v33:%.*]] = llvm.mlir.constant
    // CHECK-NEXT: [[v34:%.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK-NEXT: [[v35:%.*]] = llvm.inttoptr [[v34]] : i64 to !llvm.ptr
    // CHECK-NEXT: [[v36:%.*]] = llvm.call @MPI_recv([[v31]], [[v32]], [[v16]], [[v16]], [[v33]], [[v35]]) : (!llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32
    mpi.recv(%arg0, %rank, %rank) : memref<100xf32>, i32, i32

    // CHECK: [[v37:%.*]] = llvm.extractvalue [[v7]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK-NEXT: [[v38:%.*]] = llvm.extractvalue [[v7]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
    // CHECK-NEXT: [[v39:%.*]] = llvm.getelementptr [[v37]][[[v38]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    // CHECK-NEXT: [[v40:%.*]] = llvm.mlir.constant
    // CHECK-NEXT: [[v41:%.*]] = llvm.mlir.constant
    // CHECK-NEXT: [[v42:%.*]] = llvm.mlir.constant(1 : i64) : i64
    // CHECK-NEXT: [[v43:%.*]] = llvm.inttoptr [[v42]] : i64 to !llvm.ptr
    // CHECK-NEXT: [[v44:%.*]] = llvm.call @MPI_recv([[v39]], [[v40]], [[v16]], [[v16]], [[v41]], [[v43]]) : (!llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32
    %2 = mpi.recv(%arg0, %rank, %rank) : memref<100xf32>, i32, i32 -> !mpi.retval

    // CHECK: llvm.call @MPI_Finalize() : () -> i32
    %3 = mpi.finalize : !mpi.retval

    %4 = mpi.retval_check %retval = <MPI_SUCCESS> : i1

    %5 = mpi.error_class %0 : !mpi.retval
    return
  }
}
