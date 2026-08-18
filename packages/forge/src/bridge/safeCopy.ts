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
function isBufferDetached(buf: ArrayBufferLike): boolean {
  return (buf as any).detached === true || buf.byteLength === 0;
}

export type SafeCopyOptions = {
  retryDetached?: boolean;
  reacquire?: () => Float32Array;
};

export function ensureFloat32Array(
  input: unknown,
  options: SafeCopyOptions = {}
): Float32Array {
  if (input instanceof Float32Array) {
    if (!isBufferDetached(input.buffer)) {
      return input; // H-05: 복사 제거 — 이미 올바른 타입
    }
    if (options.retryDetached && options.reacquire) {
      const fresh = options.reacquire();
      if (!isBufferDetached(fresh.buffer)) {
        return fresh;
      }
    }
    throw new Error("WASM Memory Detached: ArrayBuffer has been detached by memory.grow.");
  }

  if (hasToJs(input)) {
    const jsView = input.toJs();
    
    if (jsView instanceof Float32Array) {
      if (!isBufferDetached(jsView.buffer)) {
        return jsView; // H-05: 복사 제거 — WASM 힙 뷰 그대로 반환
      }
      if (options.retryDetached && options.reacquire) {
        const fresh = options.reacquire();
        if (!isBufferDetached(fresh.buffer)) {
          return fresh;
        }
      }
      throw new Error("WASM Memory Detached: ArrayBuffer has been detached by memory.grow.");
    }
    
    if (jsView instanceof ArrayBuffer) {
      if (!isBufferDetached(jsView)) {
        return new Float32Array(jsView);
      }
      if (options.retryDetached && options.reacquire) {
        const fresh = options.reacquire();
        if (!isBufferDetached(fresh.buffer)) {
          return fresh;
        }
      }
      throw new Error("WASM Memory Detached: ArrayBuffer has been detached by memory.grow.");
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
