/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: SCRUM-350 High-Performance Web Worker Session Client Bridge
 *
 * WHAT: 메인 스레드에서 백그라운드 Web Worker와 통신하며 모델 로딩 및 스트리밍 생성을
 *      우아한 Promise & Callback API로 추상화한 클라이언트 브리지입니다.
 * WHY: 프론트엔드 UI 개발자가 복잡한 postMessage 이벤트 리스너를 직접 작성할 필요 없이,
 *      단 2줄의 코드로 UI 논블로킹 텍스트 생성을 통합할 수 있게 지원하기 위함입니다.
 * HOW: Worker Message Dispatcher -> Unique Request ID -> Resolve/Reject Promise Mapping -> Token Stream Hook.
 */
import { ModelLoadProgress } from '../loader/modelLoader';
import { TextGenerationProgress } from '../llm/llmTextGenerator';
export interface WorkerSessionOptions {
    workerUrl?: string;
    onProgress?: (progress: ModelLoadProgress) => void;
    useCache?: boolean;
}
export declare class WorkerSession {
    private worker;
    private reqSeq;
    private pendingResolvers;
    private streamHandlers;
    private progressHandler?;
    constructor(worker: Worker, onProgress?: (p: ModelLoadProgress) => void);
    private handleWorkerMessage;
    /**
     * 백그라운드 Worker에 자연어 텍스트 프롬프트를 전송하고 스트리밍으로 결과를 수신합니다.
     */
    prompt(text: string, options?: {
        maxNewTokens?: number;
        temperature?: number;
        topK?: number;
        topP?: number;
        backend?: 'webgpu' | 'cpu';
        onToken?: (token: string, progress: TextGenerationProgress) => void;
    }): Promise<string>;
    /**
     * 실행 중인 텍스트 생성을 즉시 중단합니다.
     */
    abort(): void;
    /**
     * Worker 스레드를 영구 종료하고 자원을 해제합니다.
     */
    terminate(): void;
}
