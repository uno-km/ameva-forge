/**
 * Created: 2026-08-12 12:14:52 +0900
 * Modified:
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *
 * shaderGuard.ts — WGSL 셰이더 보안 가드
 *
 * H-07 Fix: 화이트리스트에 모든 구현된 op 추가.
 * NH-07 Fix: 이 파일의 assertAllowedKernelName()을 graphExecutor.ts와 gpuCore.ts에서
 *   실제로 import하여 사용한다 (이전에는 데드 코드였음).
 */
/**
 * WHAT: 셰이더 내에서 사용될 식별자(함수명, 변수명)가 안전한 문자열인지 검사합니다.
 * WHY: 영숫자와 밑줄(_) 이외의 문자가 주입되어 비정상적인 코드 실행이나 컴파일 에러를 유도하는 셰이더 인젝션 공격을 예방하기 위해 존재합니다.
 * HOW: 정규 표현식(/^[a-zA-Z_][a-zA-Z0-9_]*$/)을 사용하여 문자열 패턴을 검증하고, 실패 시 AMEVAForgeSecurityError를 던집니다.
 */
export declare function assertSafeShaderIdentifier(identifier: string): void;
/**
 * WHAT: 셰이더에 주입되는 상수(숫자) 값이 유한한(finite) 숫자인지 확인합니다.
 * WHY: Infinity나 NaN과 같은 유효하지 않은 값이 셰이더 소스에 포함되어 GPU 연산 오류를 유발하는 것을 막기 위함입니다.
 * HOW: `Number.isFinite(value)`로 검사하고, 유한하지 않은 경우 예외를 발생시킵니다.
 */
export declare function assertAllowedShaderConstant(value: number): void;
/**
 * WHAT: 전달된 셰이더 소스 문자열에 동적 템플릿 리터럴 구문("${" 또는 "`")이 포함되어 있는지 검사합니다.
 * WHY: 신뢰할 수 없는 데이터가 셰이더 코드로 동적으로 삽입되는 인젝션 공격(Template Literal Injection)을 철저히 차단하기 위해 필요합니다.
 * HOW: 문자열의 `includes` 메서드를 통해 해당 패턴의 존재 여부를 검사하고, 발견될 경우 에러를 던집니다.
 */
export declare function assertStaticShaderSourceOnly(source: string): void;
/**
 * WHAT: 화이트리스트에 허용된 커널 이름들을 새롭게 등록(덮어쓰기)합니다.
 * WHY: 애플리케이션 초기화 단계 또는 플러그인 로드 시 동적으로 안전한 커널 목록을 확장하고 갱신할 수 있도록 유연성을 제공하기 위함입니다.
 * HOW: 제공된 Iterable 인터페이스(예: 배열)를 받아 새로운 Set 객체를 생성하고 `ALLOWED_KERNEL_NAMES` 변수를 갱신합니다.
 */
export declare function registerKernelNames(names: Iterable<string>): void;
/**
 * WHAT: 요청된 커널 이름이 허용된 화이트리스트(ALLOWED_KERNEL_NAMES)에 존재하는지 검사합니다.
 * WHY: 그래프 실행기(graphExecutor)나 GPU 코어 모듈이 연산을 수행하기 직전, 허가되지 않은 커널 호출을 차단하기 위해 사용됩니다.
 * HOW: `Set.has(name)` 메서드를 사용하여 포함 여부를 확인하고 없으면 보안 예외(SecurityError)를 발생시킵니다.
 */
export declare function assertAllowedKernelName(name: string): void;
/**
 * WHAT: 현재 설정된 커널 이름 화이트리스트(Set)의 읽기 전용 참조를 반환합니다.
 * WHY: 외부 모듈에서 화이트리스트의 구성을 확인할 수 있게 하면서도 직접적인 데이터 변조는 방지하기 위해 존재합니다.
 * HOW: 모듈 내부의 `ALLOWED_KERNEL_NAMES` 변수를 ReadonlySet 타입으로 캐스팅하여 그대로 반환합니다.
 */
export declare function getAllowedKernelNames(): ReadonlySet<string>;
