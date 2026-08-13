/**
 * Created: 2026-08-12 12:14:52 +0900
 * Modified:
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 * 
 * safeCopy.ts — Pyodide PyProxy → Float32Array 안전 변환
 *
 * H-05 Fix: ensureFloat32Array에서 불필요한 new Float32Array(jsView) deep copy 제거.
 *   Float32Array가 이미 WASM 힙을 가리키고 있으면 그대로 반환 (Zero-Copy).
 *   복사가 실제로 필요한 경우에만 cloneToFloat32Array()를 명시적으로 호출.
 */

/**
 * WHAT: Pyodide의 PyProxy 객체(또는 그와 유사한 구조의 객체)가 가지는 형태를 정의하는 인터페이스입니다.
 * WHY: 자바스크립트 측에서 파이썬 객체의 데이터를 가져오기 위한 `toJs()` 메서드의 존재를 명시하여 컴파일 타임에 타입 안정성을 높이기 위해 존재합니다.
 * HOW: 타입스크립트 인터페이스로 선언하여 `toJs` 속성이 반환값을 알 수 없는 함수임을 규정합니다.
 */
interface PyodideLikeProxy {
  /** WHAT: 파이썬 객체를 자바스크립트 객체로 변환하는 메서드입니다. WHY: 파이썬 데이터를 다루기 위해. HOW: 함수 호출을 통해 JS 값 반환. */
  toJs: () => unknown;
}

/**
 * WHAT: 주어진 객체가 `toJs` 메서드를 가진 PyodideLikeProxy 타입인지 런타임에 확인하는 타입 가드 함수입니다.
 * WHY: 객체의 유효성과 `toJs` 속성의 함수 여부를 동적으로 검사하여, 런타임 에러 없이 안전하게 PyProxy의 데이터를 자바스크립트 영역으로 추출할 수 있도록 보장하는 역할을 합니다.
 * HOW: typeof 연산자와 in 연산자를 사용하여 입력된 값이 객체이며 null이 아니고, 'toJs' 속성이 존재하며 그 타입이 'function'인지 논리식으로 평가하여 불리언 결과를 반환합니다.
 * 
 * @param input 검사할 임의의 데이터
 * @returns `toJs` 메서드가 존재하고 함수이면 true
 */
function hasToJs(input: unknown): input is PyodideLikeProxy {
  return (
    typeof input === "object" &&
    input !== null &&
    "toJs" in input &&
    typeof (input as { toJs?: unknown }).toJs === "function"
  );
}

/**
 * WHAT: 주어진 입력 데이터를 검증하고 안전하게 Float32Array 형태로 변환 혹은 반환하는 함수입니다. (H-05 Fix 적용)
 * WHY: 데이터의 중복 복사를 막아(Zero-Copy) 대용량 텐서 데이터 전송 시의 성능 저하를 방지하면서도, 데이터 타입의 일관성을 유지하기 위해 존재합니다.
 * HOW: 이미 Float32Array 형태이면 원본 그대로 반환합니다. PyProxy인 경우 toJs() 결과를 추출한 후 Float32Array이면 그대로 반환하고, ArrayBuffer라면 새로운 Float32Array 뷰로 래핑하여 반환합니다. 그 외에는 예외를 던집니다.
 *
 * @param input Pyodide Proxy 객체이거나 ArrayBuffer/Float32Array 형태의 데이터
 * @returns 확보된 Float32Array
 */
export function ensureFloat32Array(input: unknown): Float32Array {
  /**
   * WHAT: 입력 데이터가 Float32Array 타입인지 확인하는 조건문입니다.
   * WHY: 불필요한 변환 과정을 생략하고 즉시 반환하여 성능(Zero-Copy)을 최적화하기 위함입니다.
   * HOW: instanceof 연산자를 사용하여 입력 객체의 프로토타입 체인을 검사합니다.
   */
  if (input instanceof Float32Array) {
    return input; // H-05: 복사 제거 — 이미 올바른 타입
  }

  /**
   * WHAT: 입력 데이터가 PyodideProxy와 유사한지 확인하는 조건문입니다.
   * WHY: 파이썬으로부터 넘어온 래퍼 객체인 경우 이를 자바스크립트가 인식할 수 있는 원시 버퍼로 추출하기 위해 필요합니다.
   * HOW: 앞서 정의한 hasToJs 타입 가드 함수를 호출하여 조건을 평가합니다.
   */
  if (hasToJs(input)) {
    /**
     * WHAT: 파이썬 객체(PyProxy)로부터 자바스크립트에서 다룰 수 있는 원시 뷰(JS View)나 배열을 추출하여 담는 변수입니다.
     * WHY: 파이썬의 메모리 영역에 있는 데이터를 자바스크립트 타입 시스템 내에서 분석하고 활용하기 위해 존재합니다.
     * HOW: input.toJs() 메서드를 호출하여 반환된 값을 할당합니다.
     */
    const jsView = input.toJs();
    
    /**
     * WHAT: 추출된 뷰가 Float32Array 타입인지 확인하는 조건문입니다.
     * WHY: WASM 힙 뷰를 이미 올바른 포맷으로 갖고 있다면 또 다른 래핑 없이 바로 재사용하여 메모리 오버헤드를 막기 위함입니다.
     * HOW: instanceof Float32Array 연산자로 타입을 확인한 후 참이면 그대로 리턴합니다.
     */
    if (jsView instanceof Float32Array) {
      return jsView; // H-05: 복사 제거 — WASM 힙 뷰 그대로 반환
    }
    
    /**
     * WHAT: 추출된 뷰가 ArrayBuffer 타입인지 확인하는 조건문입니다.
     * WHY: 순수 바이트 배열인 경우 우리가 다룰 수 있는 32비트 부동소수점 데이터 뷰(Float32Array)로 해석하기 위해서입니다.
     * HOW: instanceof ArrayBuffer 연산자로 확인한 후, new Float32Array(jsView)를 호출하여 새로운 뷰를 생성하고 리턴합니다.
     */
    if (jsView instanceof ArrayBuffer) {
      return new Float32Array(jsView);
    }
  }

  throw new Error(
    "Invalid input type: expected Float32Array or a Pyodide proxy coercible to Float32Array."
  );
}

/**
 * WHAT: 입력 데이터를 강제로 새로운 메모리 공간에 깊은 복사(Deep Copy)하여 반환하는 함수입니다. (명시적 deep copy 용도)
 * WHY: 원본 데이터(WASM 힙 등)가 삭제되거나 변경되어도 안전하게 데이터를 보존해야 할 때, 혹은 독립적인 데이터 소유권을 가지는 버퍼가 필요할 때 호출하기 위해 존재합니다. 일반 데이터 읽기에는 성능상 사용하지 않아야 합니다.
 * HOW: ensureFloat32Array를 호출하여 먼저 안전한 뷰를 확보한 뒤, new Float32Array(view)를 사용하여 동일한 요소들을 가지는 완전히 새로운 메모리 배열 인스턴스를 할당하여 반환합니다.
 * 
 * @param input 원본 데이터
 * @returns 독립된 메모리 공간을 가지는 복사된 Float32Array
 */
export function cloneToFloat32Array(input: unknown): Float32Array {
  /**
   * WHAT: 원본 데이터로부터 읽기 가능한 Float32Array 뷰를 안전하게 가져와 담아두는 변수입니다.
   * WHY: 복사를 수행하기 전, 원본 데이터가 어떤 형태이든(PyProxy, ArrayBuffer 등) 통일된 Float32Array 포맷으로 만들어놓기 위해서입니다.
   * HOW: ensureFloat32Array(input) 함수를 호출하여 반환값을 저장합니다.
   */
  const view = ensureFloat32Array(input);
  return new Float32Array(view);
}
