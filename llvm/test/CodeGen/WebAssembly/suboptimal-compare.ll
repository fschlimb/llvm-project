; RUN: llc < %s -fast-isel -O0 | FileCheck %s

target triple = "wasm32-unknown-unknown"

; CHECK-LABEL: gh_80053:                               # @gh_80053
; CHECK-NEXT: .functype	gh_80053 (i32) -> (i32)
; CHECK:      block
; CHECK-NEXT: local.get	0
; CHECK-NEXT: i32.const	0
; CHECK-NEXT: i32.eq  
; CHECK-NEXT: i32.const	1
; CHECK-NEXT: i32.and 
; CHECK-NEXT: i32.eqz
; CHECK-NEXT: br_if   	0                               # 0: down to label0
; CHECK:      i32.const	0
; CHECK-NEXT: return
; CHECK-NEXT: .LBB0_2:                                # %BB03
; CHECK-NEXT: end_block                               # label0:
; CHECK-NEXT: i32.const	1
; CHECK-NEXT: return
; CHECK-NEXT: end_function
define i1 @gh_80053(ptr) {
BB01:
    %eq = icmp eq ptr %0, null
    br i1 %eq, label %BB02, label %BB03
BB02:
    ret i1 0
BB03:
    ret i1 1
}
