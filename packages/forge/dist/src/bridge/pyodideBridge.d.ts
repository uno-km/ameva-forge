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
 * WHAT: Pyodide가 자바스크립트 기능에 접근할 수 있도록 전역 `globalThis.amevaForge` 객체를 생성하고 등록합니다.
 * WHY: 파이썬 측 브리지 코드가 WASM을 거쳐 GPU 하드웨어 가속(WebGPU 등) 기능과 그래프 실행 로직을 사용할 수 있게 하는 엔트리 포인트가 필요하기 때문입니다.
 * HOW: 필요한 모든 내부 함수들을 모은 api 객체를 만들고 Object.freeze로 동결시킨 뒤, globalThis의 속성으로 할당하여 전역에서 접근 가능하게 만듭니다.
 *
 * @returns 등록된 전역 API 객체
 */
export declare function registerPyodideBridge(): AmevaTensorGlobalAPI;
