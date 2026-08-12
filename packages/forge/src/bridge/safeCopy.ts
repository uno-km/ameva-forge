/**
 * safeCopy.ts — Pyodide PyProxy → Float32Array 안전 변환
 *
 * H-05 Fix: ensureFloat32Array에서 불필요한 new Float32Array(jsView) deep copy 제거.
 *   Float32Array가 이미 WASM 힙을 가리키고 있으면 그대로 반환 (Zero-Copy).
 *   복사가 실제로 필요한 경우에만 cloneToFloat32Array()를 명시적으로 호출.
 */

interface PyodideLikeProxy {
  toJs: () => unknown;
}

function hasToJs(input: unknown): input is PyodideLikeProxy {
  return (
    typeof input === "object" &&
    input !== null &&
    "toJs" in input &&
    typeof (input as { toJs?: unknown }).toJs === "function"
  );
}

/**
 * H-05 Fix: 입력이 이미 Float32Array면 복사 없이 반환.
 * PyProxy인 경우 toJs() 결과를 확인하고 역시 복사 없이 뷰를 반환.
 */
export function ensureFloat32Array(input: unknown): Float32Array {
  if (input instanceof Float32Array) {
    return input; // H-05: 복사 제거 — 이미 올바른 타입
  }

  if (hasToJs(input)) {
    const jsView = input.toJs();
    if (jsView instanceof Float32Array) {
      return jsView; // H-05: 복사 제거 — WASM 힙 뷰 그대로 반환
    }
    if (jsView instanceof ArrayBuffer) {
      return new Float32Array(jsView);
    }
  }

  throw new Error(
    "Invalid input type: expected Float32Array or a Pyodide proxy coercible to Float32Array."
  );
}

/**
 * 명시적 deep copy가 필요한 경우 (예: 버퍼 소유권 이전).
 * 일반 데이터 읽기에는 사용하지 말 것.
 */
export function cloneToFloat32Array(input: unknown): Float32Array {
  const view = ensureFloat32Array(input);
  return new Float32Array(view);
}
