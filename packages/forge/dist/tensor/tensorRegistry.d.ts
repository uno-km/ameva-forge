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
/**
 * WHAT: GPU 텐서의 생명주기를 관리하는 레지스트리 클래스입니다.
 * WHY: 생성된 텐서의 메타데이터와 WebGPU 버퍼를 중앙에서 추적하고 메모리 누수를 방지하기 위해 존재합니다.
 * HOW: Map 객체를 사용하여 고유한 핸들(TensorHandle)을 키로, 텐서 레코드(TensorRecord)를 값으로 저장 및 관리합니다.
 */
export declare class TensorRegistry {
    private records;
    private nextId;
    snapshotHandles(): string[];
    registerRecord(record: Omit<TensorRecord, 'createdAt' | 'disposed'>): TensorHandle;
    /**
     * WHAT: 새로운 텐서를 레지스트리에 등록하고 고유 핸들을 반환하는 함수입니다.
     * WHY: WebGPU 버퍼 및 메타데이터를 프레임워크가 추적할 수 있도록 레지스트리에 기록하기 위함입니다.
     * HOW: 예측 불가능한 UUID 기반의 핸들을 생성하고, 입력받은 정보와 함께 내부 records 맵에 저장합니다.
     */
    register(recordOmitHandle: Omit<TensorRecord, 'handle' | 'disposed' | 'createdAt'>): TensorHandle;
    /**
     * WHAT: 주어진 핸들을 사용하여 텐서 레코드를 조회하는 함수입니다.
     * WHY: 연산을 수행할 때 필요한 텐서의 메타데이터와 실제 WebGPU 버퍼를 가져오기 위해 존재합니다.
     * HOW: 내부 records 맵에서 핸들을 키로 조회하며, 존재하지 않거나 이미 폐기된 경우 에러를 발생시킵니다.
     */
    get(handle: TensorHandle): TensorRecord;
    /**
     * WHAT: 특정 핸들의 텐서가 유효하게 존재하는지 확인하는 함수입니다.
     * WHY: 텐서가 해제(dispose)되었는지 예외 없이 안전하게 체크하기 위해 사용됩니다.
     * HOW: 핸들로 레코드를 조회하여 undefined가 아니고 disposed 상태가 아닌지(boolean)를 반환합니다.
     */
    has(handle: TensorHandle): boolean;
    /**
     * WHAT: 지정된 핸들의 텐서를 폐기하고 GPU 메모리를 해제하는 함수입니다.
     * WHY: 사용이 끝난 텐서의 메모리를 반환하여 OOM(Out of Memory)을 방지하고 자원 누수를 막기 위함입니다.
     * HOW: 레코드를 disposed로 표시하고 맵에서 제거한 뒤, QuotaManager와 WebGPU 큐를 통해 버퍼를 해제합니다.
     */
    dispose(handle: TensorHandle): void;
    markFailed(handle: TensorHandle, errorMsg: string): void;
    /**
     * WHAT: 레지스트리에 등록된 모든 텐서를 일괄 폐기하는 함수입니다.
     * WHY: 컨텍스트 초기화나 애플리케이션 종료 시 모든 GPU 자원을 확실하게 정리하기 위해 존재합니다.
     * HOW: 아직 폐기되지 않은 모든 레코드를 수집하고, 맵을 비운 뒤 GPU 큐가 비워지면 버퍼를 순차적으로 해제합니다.
     */
    clear(): void;
}
/**
 * WHAT: 전역적으로 사용되는 텐서 레지스트리의 단일 인스턴스(싱글톤)입니다.
 * WHY: 애플리케이션 전체에서 동일한 텐서 관리 상태를 공유하기 위해 생성됩니다.
 * HOW: TensorRegistry 클래스의 새 인스턴스를 생성하여 내보냅니다(export).
 */
export declare const _globalRegistry: TensorRegistry;
