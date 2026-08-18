/**
 * Created: 2026-08-12 12:14:52 +0900
 * Modified:
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 * 
 * pyodideBridge.ts — globalThis.amevaForge API 등록자
 *
 * H-02 연동: 단일 실행 경로(graphExecutor.ts)로 통합.
 *   executeGraph 시그니처: (instructionsJson: string, jsInputs: unknown) => Record
 *
 * M-06 연동: disposeBatch 추가 (bridge.py의 js_dispose_batch 지원)
 */

import { init, read, dispose, getTensorInfo, mapBufferAsync, readMappedInto, warmupKernels } from "../tensor/gpuCore";
import { executeGraph } from "../tensor/graphExecutor";
import { _globalRegistry } from "../tensor/tensorRegistry";
import { getQuotaSnapshot } from "../webgpu/quota";
import { getDevice, _safeLog } from "../webgpu/device";
import { clearStagingPool } from "../webgpu/buffers";
import { _globalUniformPool } from "../webgpu/uniformPool";
import { TensorHandle } from "../types";

/**
 * WHAT: 이 인터페이스는 전역 amevaForge 객체의 형태를 정의합니다.
 * WHY: 이 API의 목적은 파이오다이드(Pyodide) 환경의 파이썬 코드에서
 *      자바스크립트/웹어셈블리(WASM) 쪽의 GPU 핵심 기능과 그래프 실행 기능을 호출할 수 있도록 
 *      타입스크립트 브리지(bridge) 역할을 하는 것입니다.
 * HOW: 이 인터페이스를 통해 파이썬이 GPU 메모리 관리 및 연산 실행 관련 함수들에 접근하여
 *      WebGPU 자원을 다룰 수 있도록 구조화합니다.
 */
export interface AmevaTensorGlobalAPI {
  /** WHAT: GPU 코어 초기화 함수. WHY: WebGPU 디바이스를 준비하기 위해. HOW: WebGPU API를 호출해 설정. */
  init: typeof init;
  /** WHAT: 텐서 데이터 읽기 함수. WHY: GPU 메모리 데이터를 메인 메모리로 가져오기 위해. HOW: 비동기로 버퍼 매핑 후 데이터 복사. */
  read: typeof read;
  /** WHAT: 텐서 메모리 해제 함수. WHY: 사용이 끝난 GPU 자원을 반환하기 위해. HOW: WebGPU 버퍼의 destroy 메서드 호출. */
  dispose: typeof dispose;
  /** WHAT: 텐서 메타데이터 조회 함수. WHY: 텐서의 크기, 타입, 상태를 확인하기 위해. HOW: 내부 레지스트리에서 정보 조회. */
  getTensorInfo: typeof getTensorInfo;
  /** WHAT: 비동기 버퍼 매핑 함수. WHY: 데이터를 효율적으로 읽기 위해 매핑 상태로 만들기 위함. HOW: mapAsync를 호출. */
  mapBufferAsync: typeof mapBufferAsync;
  /** WHAT: 매핑된 버퍼를 특정 타입 배열로 읽어오는 함수. WHY: 복사 오버헤드 없이 직접 뷰를 가져오기 위함. HOW: getMappedRange 결과를 TypedArray로 변환. */
  readMappedInto: typeof readMappedInto;
  /** WHAT: 텐서 연산 그래프 실행 함수. WHY: 복잡한 연산들을 순차적으로 GPU에서 수행하기 위함. HOW: JSON 명령어 파싱 후 각 커널 실행. */
  executeGraph: typeof executeGraph;
  /** WHAT: 커널 웜업 함수. WHY: 런타임 성능을 안정화하기 위해 미리 셰이더를 컴파일하기 위함. HOW: 파이프라인을 미리 생성. */
  warmupKernels: typeof warmupKernels;
  
  /** 
   * WHAT: M-06 batch dispose — 여러 텐서 핸들 배열을 한 번에 해제.
   * WHY: 파이썬 쪽에서 여러 개의 텐서를 가비지 컬렉션할 때 단일 호출로 성능을 높이기 위해.
   * HOW: 전달된 배열을 순회하며 개별 dispose를 호출.
   */
  disposeBatch: (handles: TensorHandle[]) => void;
  getQuotaSnapshot: typeof getQuotaSnapshot;
  snapshotHandles: () => string[];
  flushGC: (options?: { strict?: boolean }) => Promise<{ ok: boolean; error?: string }>;
}

declare global {
  /**
   * WHAT: 전역 네임스페이스(globalThis)에 amevaForge 객체를 등록하기 위한 선언입니다.
   * WHY: 브라우저나 워커 환경 어디서든 전역 스코프에서 이 브리지 객체에 접근할 수 있게 하기 위해 존재합니다.
   * HOW: var 키워드를 통해 전역 타입 확장을 수행합니다.
   */
  var amevaForge: AmevaTensorGlobalAPI | undefined;
}

/**
 * WHAT: 여러 개의 텐서 핸들(TensorHandle)을 한 번에 일괄적으로 메모리에서 해제(dispose)합니다.
 * WHY: 파이썬(Pyodide) 환경에서 다수의 텐서 가비지 컬렉션을 효율적으로 처리하기 위해 존재합니다. (단일 호출로 오버헤드 감소)
 * HOW: 반복문(for...of)을 통해 각 핸들마다 GPU 메모리 해제를 시도하며, 이미 해제된 텐서의 에러는 조용히 무시하여 중단되지 않도록 처리합니다.
 * 
 * @param handles 해제할 텐서 핸들들의 배열
 */
function disposeBatch(handles: TensorHandle[]): void {
  /**
   * WHAT: 입력받은 핸들 배열을 순회하는 반복문입니다.
   * WHY: 각각의 텐서 리소스에 대해 개별적인 해제 절차가 필요하기 때문에 존재합니다.
   * HOW: for...of 구문을 사용하여 handles 배열의 각 원소(handle)를 하나씩 가져와 내부 블록을 실행합니다.
   */
  for (const handle of handles) {
    /** 
     * WHAT: 현재 순회 중인 텐서 핸들이 유효한 값(truthy)인지 확인하는 조건문입니다.
     * WHY: null, undefined 혹은 빈 문자열 같은 잘못된 핸들이 전달되어 불필요한 예외나 시스템 오류가 발생하는 것을 방지하기 위함입니다.
     * HOW: 자바스크립트의 truthy 평가를 통해 handle 값이 존재할 때만 내부의 해제 로직(try-catch 블록)을 수행하도록 제어합니다.
     */
    if (handle) {
      try { 
        dispose(handle); 
      } catch (e) { 
        _safeLog(`[pyodideBridge] disposeBatch handle "${handle}" failed: ${e}`);
      }
    }
  }
}

/**
 * WHAT: Pyodide가 자바스크립트 기능에 접근할 수 있도록 전역 `globalThis.amevaForge` 객체를 생성하고 등록합니다.
 * WHY: 파이썬 측 브리지 코드가 WASM을 거쳐 GPU 하드웨어 가속(WebGPU 등) 기능과 그래프 실행 로직을 사용할 수 있게 하는 엔트리 포인트가 필요하기 때문입니다.
 * HOW: 필요한 모든 내부 함수들을 모은 api 객체를 만들고 Object.freeze로 동결시킨 뒤, globalThis의 속성으로 할당하여 전역에서 접근 가능하게 만듭니다.
 * 
 * @returns 등록된 전역 API 객체
 */
export function registerPyodideBridge(): AmevaTensorGlobalAPI {
  /**
   * WHAT: 실제로 전역에 노출될 API 객체를 구성하는 변수입니다.
   * WHY: 각 기능(init, read 등)들을 하나의 통일된 인터페이스 객체로 모아서 파이썬 측에서 구조화된 방식으로 쉽게 접근할 수 있게 묶어주기 위함입니다.
   * HOW: AmevaTensorGlobalAPI 타입에 맞추어 내부 모듈에서 임포트한 함수들을 프로퍼티로 할당하여 객체 리터럴을 생성합니다.
   */
  const api: AmevaTensorGlobalAPI = {
    init,
    read,
    dispose,
    getTensorInfo,
    mapBufferAsync,
    readMappedInto,
    executeGraph,
    warmupKernels,
    disposeBatch,
    getQuotaSnapshot,
    snapshotHandles: () => _globalRegistry.snapshotHandles(),
    flushGC: async (options?: { strict?: boolean }) => {
      try {
        const dev = getDevice();
        await dev.queue.onSubmittedWorkDone();
        await _globalUniformPool.retireSubmitted(dev);
        clearStagingPool();
        _globalUniformPool.clear();
        return { ok: true };
      } catch (e: any) {
        _safeLog(`[pyodideBridge] flushGC work done error: ${e}`);
        if (options && options.strict) {
          throw e;
        }
        return { ok: false, error: String(e) };
      }
    },
  };

  Object.freeze(api); // F-014 Fix: API 객체 동결하여 외부 변조 방지
  globalThis.amevaForge = api;
  return api;
}
