/**
 * shaderGuard.ts — WGSL 셰이더 보안 가드
 *
 * H-07 Fix: 화이트리스트에 모든 구현된 op 추가.
 * NH-07 Fix: 이 파일의 assertAllowedKernelName()을 graphExecutor.ts와 gpuCore.ts에서
 *   실제로 import하여 사용한다 (이전에는 데드 코드였음).
 */

import { AMEVAForgeSecurityError } from "../errors";

/** 셰이더 식별자 (함수명, 변수명 등) 유효성 검사 */
export function assertSafeShaderIdentifier(identifier: string): void {
  if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(identifier)) {
    throw new AMEVAForgeSecurityError(
      `Invalid shader identifier: "${identifier}". Only alphanumeric and underscore allowed.`
    );
  }
}

/** 셰이더에 삽입되는 상수값 유효성 검사 */
export function assertAllowedShaderConstant(value: number): void {
  if (!Number.isFinite(value)) {
    throw new AMEVAForgeSecurityError(
      `Invalid shader constant: ${value}. Must be a finite number.`
    );
  }
}

/**
 * 동적 문자열 보간이 셰이더 소스에 포함되지 않았는지 검사.
 * 템플릿 리터럴 인젝션 공격을 차단한다.
 */
export function assertStaticShaderSourceOnly(source: string): void {
  if (source.includes("${") || source.includes("`")) {
    throw new AMEVAForgeSecurityError(
      "Dynamic shader source interpolation is forbidden. Use uniform buffers for runtime values."
    );
  }
}

/**
 * H-07/NH-07 Fix: 모든 구현된 커널 이름을 화이트리스트에 포함.
 * graphExecutor.ts의 ALLOWED_OPS와 반드시 동기화 유지.
 * 이 함수는 gpuCore.ts와 graphExecutor.ts에서 실제로 호출된다.
 */
let ALLOWED_KERNEL_NAMES = new Set([
  "matmul",
  "relu",
  "relu_backward",
  "add",
  "mul",
  "transpose",
  // v2.0: 학습 기능에 필요한 커널 추가 (VUL-001 Fix)
  "sub",
  "neg",
  "div",
  "exp",
  "log",
  "sigmoid",
  "tanh",
  "sigmoid_backward",
  "tanh_backward",
  "fill",
  "sum",
  "max",
  "sum_axis",
  "axpy",
  "pad",
  "gather",
  "scatter",
]);

export function registerKernelNames(names: Iterable<string>): void {
  ALLOWED_KERNEL_NAMES = new Set(names);
}

export function assertAllowedKernelName(name: string): void {
  if (!ALLOWED_KERNEL_NAMES.has(name)) {
    throw new AMEVAForgeSecurityError(
      `Unknown kernel name: "${name}". Allowed: ${[...ALLOWED_KERNEL_NAMES].join(", ")}`
    );
  }
}

/** 허용된 커널 이름 목록 반환 (외부 동기화 용도) */
export function getAllowedKernelNames(): ReadonlySet<string> {
  return ALLOWED_KERNEL_NAMES;
}
