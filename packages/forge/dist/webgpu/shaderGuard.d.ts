/**
 * shaderGuard.ts — WGSL 셰이더 보안 가드
 *
 * H-07 Fix: 화이트리스트에 모든 구현된 op 추가.
 * NH-07 Fix: 이 파일의 assertAllowedKernelName()을 graphExecutor.ts와 gpuCore.ts에서
 *   실제로 import하여 사용한다 (이전에는 데드 코드였음).
 */
/** 셰이더 식별자 (함수명, 변수명 등) 유효성 검사 */
export declare function assertSafeShaderIdentifier(identifier: string): void;
/** 셰이더에 삽입되는 상수값 유효성 검사 */
export declare function assertAllowedShaderConstant(value: number): void;
/**
 * 동적 문자열 보간이 셰이더 소스에 포함되지 않았는지 검사.
 * 템플릿 리터럴 인젝션 공격을 차단한다.
 */
export declare function assertStaticShaderSourceOnly(source: string): void;
export declare function registerKernelNames(names: Iterable<string>): void;
export declare function assertAllowedKernelName(name: string): void;
/** 허용된 커널 이름 목록 반환 (외부 동기화 용도) */
export declare function getAllowedKernelNames(): ReadonlySet<string>;
