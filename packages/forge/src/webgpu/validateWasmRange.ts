/**
 * Created: 2026-08-12 12:14:52 +0900
 * Modified:
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 * 
 * WHAT: WASM(WebAssembly) 메모리 영역에 접근할 때 오프셋과 길이의 유효성을 검증하는 모듈입니다.
 * WHY: 잘못된 메모리 주소나 범위를 참조하여 발생하는 버퍼 오버플로우, 세그멘테이션 폴트 및 잠재적 보안 취약점을 차단하기 위해 필요합니다.
 */
import { AMEVAForgeSecurityError } from "../errors";

/**
 * WHAT: 주어진 오프셋과 데이터 길이가 WASM 선형 메모리 힙(heap)의 유효한 범위 내에 있는지 안전하게 검사합니다.
 * WHY: CPU-GPU 간 데이터 전송이나 공유 메모리 접근 시 악의적이거나 잘못된 크기 요청으로 인한 메모리 침범을 방어하기 위해 호출됩니다.
 * HOW: `Number.isSafeInteger`와 비음수(non-negative) 조건을 통해 입력 인자의 데이터 타입을 엄격히 검증한 후, `offset + byteLength`가 총 WASM 메모리 크기를 초과하지 않는지 계산하여 확인합니다. 위반 시 보안 예외를 던집니다.
 */
export function assertWasmRange(offset: number, byteLength: number, wasmByteLength: number): void {
  if (!Number.isSafeInteger(offset) || offset < 0) {
    throw new AMEVAForgeSecurityError("Invalid offset: must be a non-negative safe integer.");
  }
  if (!Number.isSafeInteger(byteLength) || byteLength < 0) {
    throw new AMEVAForgeSecurityError("Invalid byteLength: must be a non-negative safe integer.");
  }
  if (!Number.isSafeInteger(wasmByteLength) || wasmByteLength < 0) {
    throw new AMEVAForgeSecurityError("Invalid wasmByteLength: must be a non-negative safe integer.");
  }

  if (offset > wasmByteLength || byteLength > wasmByteLength - offset) {
    throw new AMEVAForgeSecurityError("WASM memory range out of bounds");
  }
}
