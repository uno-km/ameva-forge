/**
 * Created: 2026-08-12T12:14:52+09:00
 * Modified:
 *   - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *
 * tensorRegistry.ts — GPU 텐서 생명주기 레지스트리
 *
 * C-06 Fix: dispose() 시 _globalQuotaManager.markPendingRelease() 즉시 호출.
 * NC-07 Fix: dynamic import() 제거 → 정적 import 사용 + device.destroy() 보장.
 * NL-03 Fix: Date.now() 제거 → 단조증가 ID만 사용 (타이밍 정보 노출 방지).
 */

import { TensorHandle, TensorRecord } from "../types";
import { AMEVAForgeDisposedError } from "../errors";
import { freeBuffer } from "../webgpu/buffers";
import { _globalQuotaManager } from "../webgpu/quota";
import { getDevice } from "../webgpu/device";

/**
 * WHAT: GPU 텐서의 생명주기를 관리하는 레지스트리 클래스입니다.
 * WHY: 생성된 텐서의 메타데이터와 WebGPU 버퍼를 중앙에서 추적하고 메모리 누수를 방지하기 위해 존재합니다.
 * HOW: Map 객체를 사용하여 고유한 핸들(TensorHandle)을 키로, 텐서 레코드(TensorRecord)를 값으로 저장 및 관리합니다.
 */
class TensorRegistry {
  /**
   * WHAT: 텐서 핸들과 텐서 레코드를 매핑하여 저장하는 내부 상태 변수입니다.
   * WHY: 생성된 모든 텐서에 빠르게 접근하고 상태를 업데이트하기 위해 해시맵(Map)을 사용합니다.
   * HOW: TensorHandle(문자열)을 키로, TensorRecord 객체를 값으로 유지합니다.
   */
  private records: Map<TensorHandle, TensorRecord> = new Map();

  /**
   * WHAT: 다음에 생성될 텐서에 부여될 단조 증가 식별자입니다.
   * WHY: 타이밍 정보 노출(부채널 공격)을 방지하기 위해 Date.now() 대신 단순 증가 ID를 사용합니다.
   * HOW: 텐서가 새로 등록될 때마다 1씩 증가하여 각 텐서 레코드의 createdAt 필드에 할당됩니다.
   */
  private nextId: number = 1;

  /**
   * WHAT: 새로운 텐서를 레지스트리에 등록하고 고유 핸들을 반환하는 함수입니다.
   * WHY: WebGPU 버퍼 및 메타데이터를 프레임워크가 추적할 수 있도록 레지스트리에 기록하기 위함입니다.
   * HOW: 예측 불가능한 UUID 기반의 핸들을 생성하고, 입력받은 정보와 함께 내부 records 맵에 저장합니다.
   */
  register(
    recordOmitHandle: Omit<TensorRecord, 'handle' | 'disposed' | 'createdAt'>
  ): TensorHandle {
    // F-015 Fix: 예측 가능한 핸들 생성을 막기 위해 암호학적 난수 기반 식별자 사용
    /**
     * WHAT: 암호학적으로 안전한 무작위 식별자(UUID) 문자열입니다.
     * WHY: 악의적인 사용자가 다른 텐서의 핸들을 추측하여 접근하는 것을 방지하기 위해 생성됩니다.
     * HOW: crypto.randomUUID가 사용 가능하면 이를 호출하고, 그렇지 않으면 Math.random()을 기반으로 임시 문자열을 생성합니다.
     */
    const uuid = typeof crypto !== 'undefined' && crypto.randomUUID 
      ? crypto.randomUUID() 
      : Math.random().toString(36).substring(2, 15);
    
    /**
     * WHAT: 텐서를 고유하게 식별하기 위한 최종 핸들 문자열입니다.
     * WHY: 외부에서 텐서를 참조할 때 이 문자열을 사용하여 안전하게 접근할 수 있도록 제공됩니다.
     * HOW: "tensor_" 접두사와 위에서 생성한 uuid 문자열을 결합하여 생성됩니다.
     */
    const handle = `tensor_${uuid}`;
    
    /**
     * WHAT: 레지스트리에 저장될 텐서의 모든 메타데이터를 포함하는 레코드 객체입니다.
     * WHY: WebGPU 버퍼, 모양(shape), 데이터 타입, 생성 순서 등을 한 곳에서 관리하기 위함입니다.
     * HOW: 전달된 recordOmitHandle 객체에 handle, disposed=false, createdAt(단조증가 ID)을 병합하여 생성합니다.
     */
    const record: TensorRecord = {
      ...recordOmitHandle,
      handle,
      disposed: false,
      createdAt: this.nextId - 1  // NL-03 Fix: monotonic counter, not timestamp
    };
    
    this.records.set(handle, record);
    this.nextId++;
    return handle;
  }

  /**
   * WHAT: 주어진 핸들을 사용하여 텐서 레코드를 조회하는 함수입니다.
   * WHY: 연산을 수행할 때 필요한 텐서의 메타데이터와 실제 WebGPU 버퍼를 가져오기 위해 존재합니다.
   * HOW: 내부 records 맵에서 핸들을 키로 조회하며, 존재하지 않거나 이미 폐기된 경우 에러를 발생시킵니다.
   */
  get(handle: TensorHandle): TensorRecord {
    /**
     * WHAT: 레지스트리에서 핸들로 조회한 텐서 레코드입니다.
     * WHY: 텐서가 실제로 존재하는지 검증하고 접근하기 위해 임시 변수에 저장합니다.
     * HOW: this.records.get(handle)을 통해 값을 가져옵니다.
     */
    const record = this.records.get(handle);
    
    if (!record) {
      throw new AMEVAForgeDisposedError(`Tensor not found: ${handle}`);
    }
    if (record.disposed) {
      throw new AMEVAForgeDisposedError(`Attempted to access disposed tensor: ${handle}`);
    }
    return record;
  }

  /**
   * WHAT: 특정 핸들의 텐서가 유효하게 존재하는지 확인하는 함수입니다.
   * WHY: 텐서가 해제(dispose)되었는지 예외 없이 안전하게 체크하기 위해 사용됩니다.
   * HOW: 핸들로 레코드를 조회하여 undefined가 아니고 disposed 상태가 아닌지(boolean)를 반환합니다.
   */
  has(handle: TensorHandle): boolean {
    /**
     * WHAT: 조회된 텐서 레코드 변수입니다.
     * WHY: 존재 여부 및 disposed 상태를 판별하기 위해 사용합니다.
     * HOW: records 맵에서 핸들로 가져옵니다.
     */
    const record = this.records.get(handle);
    return record !== undefined && !record.disposed;
  }

  /**
   * WHAT: 지정된 핸들의 텐서를 폐기하고 GPU 메모리를 해제하는 함수입니다.
   * WHY: 사용이 끝난 텐서의 메모리를 반환하여 OOM(Out of Memory)을 방지하고 자원 누수를 막기 위함입니다.
   * HOW: 레코드를 disposed로 표시하고 맵에서 제거한 뒤, QuotaManager와 WebGPU 큐를 통해 버퍼를 해제합니다.
   */
  dispose(handle: TensorHandle): void {
    if (!this.records.has(handle)) {
        return; // TS-H04: 이중 dispose 방어 — 이미 해제된 핸들 무시
    }
    
    /**
     * WHAT: 폐기할 텐서의 레코드 객체입니다.
     * WHY: 텐서의 WebGPU 버퍼와 할당 토큰(token)에 접근하여 실제 메모리를 해제하기 위해 필요합니다.
     * HOW: this.records.get()으로 가져오며, 유효하지 않으면 조기 반환(return)합니다.
     */
    const record = this.records.get(handle);
    if (!record || record.disposed) return;

    record.disposed = true;
    this.records.delete(handle);

    // C-06 Fix: 즉시 "해제 예약" 표시
    _globalQuotaManager.markPendingRelease(record.token);

    // NC-07 Fix: 정적 import된 getDevice() 사용 (dynamic import 제거)
    try {
      /**
       * WHAT: 현재 활성화된 WebGPU 디바이스 인스턴스입니다.
       * WHY: GPU에 제출된 모든 명령이 끝난 후 안전하게 버퍼를 파괴하기 위해 필요합니다.
       * HOW: getDevice() 유틸리티 함수를 호출하여 가져옵니다.
       */
      const device = getDevice();
      device.queue.onSubmittedWorkDone().then(() => {
        freeBuffer(record.buffer, record.token);
      }).catch(() => {
        // GPU 큐 실패 → 즉시 소각
        _safeDestroyBuffer(record);
      });
    } catch {
      // device가 없거나 lost → 즉시 quota 해제 + buffer 소각
      _safeDestroyBuffer(record);
    }
  }

  // F-016 Fix: 비동기 에러 발생 시 해당 핸들에 에러를 마킹
  markFailed(handle: TensorHandle, errorMsg: string): void {
    const record = this.records.get(handle);
    if (record && !record.disposed) {
      record.error = errorMsg;
    }
  }

  /**
   * WHAT: 레지스트리에 등록된 모든 텐서를 일괄 폐기하는 함수입니다.
   * WHY: 컨텍스트 초기화나 애플리케이션 종료 시 모든 GPU 자원을 확실하게 정리하기 위해 존재합니다.
   * HOW: 아직 폐기되지 않은 모든 레코드를 수집하고, 맵을 비운 뒤 GPU 큐가 비워지면 버퍼를 순차적으로 해제합니다.
   */
  clear(): void {
    /**
     * WHAT: 아직 해제되지 않아 메모리를 점유하고 있는 텐서 레코드들의 배열입니다.
     * WHY: 맵(Map)이 초기화된 후에도 이 객체들의 버퍼를 파괴하기 위해 참조를 유지해야 합니다.
     * HOW: this.records.values()를 배열로 변환하고 disposed가 false인 것만 필터링합니다.
     */
    const recordsToFree = Array.from(this.records.values()).filter(r => !r.disposed);
    this.records.clear();

    if (recordsToFree.length === 0) return;

    /**
     * WHAT: 해제 대상 텐서들을 순회하는 루프입니다.
     * WHY: 모든 할당된 텐서에 대해 해제 대기 상태임을 QuotaManager에 알리기 위함입니다.
     * HOW: for...of 구문을 사용하여 recordsToFree 배열을 순회합니다.
     */
    for (const record of recordsToFree) {
      _globalQuotaManager.markPendingRelease(record.token);
    }

    try {
      /**
       * WHAT: WebGPU 명령 큐의 상태를 확인하기 위한 디바이스 객체입니다.
       * WHY: 큐에 대기 중인 작업이 텐서를 참조할 수 있으므로, 작업 완료 후 안전하게 해제하기 위해 사용됩니다.
       * HOW: getDevice()를 통해 인스턴스를 얻어옵니다.
       */
      const device = getDevice();
      device.queue.onSubmittedWorkDone().then(() => {
        /**
         * WHAT: GPU 작업이 완료된 후 각 버퍼를 해제하는 루프입니다.
         * WHY: 실제 VRAM과 QuotaManager의 할당량을 반환하기 위해 필요합니다.
         * HOW: for...of 루프를 돌며 freeBuffer를 호출합니다.
         */
        for (const record of recordsToFree) {
          freeBuffer(record.buffer, record.token);
        }
      }).catch(() => {
        /**
         * WHAT: GPU 큐 대기 실패 시 강제 해제하는 루프입니다.
         * WHY: 큐에러가 발생해도 메모리 누수를 방지하기 위해 존재합니다.
         * HOW: _safeDestroyBuffer 헬퍼를 직접 호출합니다.
         */
        for (const record of recordsToFree) {
          _safeDestroyBuffer(record);
        }
      });
    } catch {
      // device already lost
      /**
       * WHAT: 디바이스 유실 시 텐서 버퍼를 강제 해제하는 루프입니다.
       * WHY: 디바이스가 유실되어 큐 대기를 할 수 없으므로 남은 리소스를 정리하기 위해 필요합니다.
       * HOW: for...of 루프를 돌며 _safeDestroyBuffer를 호출하고, 이후 쿼터를 초기화합니다.
       */
      for (const record of recordsToFree) {
        _safeDestroyBuffer(record);
      }
      _globalQuotaManager.reset();
    }
  }
}

/**
 * WHAT: 텐서의 WebGPU 버퍼를 파괴하고 할당 토큰을 해제하는 유틸리티 함수입니다.
 * WHY: 디바이스 유실이나 큐 실패 상황에서도 예외 발생 없이 버퍼 자원을 반환하기 위해 존재합니다.
 * HOW: try-catch 블록 안에서 buffer.destroy()를 호출하고, _globalQuotaManager.releaseToken()을 호출합니다.
 */
function _safeDestroyBuffer(record: TensorRecord): void {
  try {
    record.buffer.destroy();
  } catch {
    // buffer가 이미 destroyed
  }
  _globalQuotaManager.releaseToken(record.token);
}

/**
 * WHAT: 전역적으로 사용되는 텐서 레지스트리의 단일 인스턴스(싱글톤)입니다.
 * WHY: 애플리케이션 전체에서 동일한 텐서 관리 상태를 공유하기 위해 생성됩니다.
 * HOW: TensorRegistry 클래스의 새 인스턴스를 생성하여 내보냅니다(export).
 */
export const _globalRegistry = new TensorRegistry();
