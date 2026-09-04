/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: SCRUM-349 Web Worker Background Neural Runner
 *
 * WHAT: 브라우저 메인 스레드와 완전히 분리된 전용 Web Worker 컨텍스트에서
 *      WebGPU 초기화, 모델 로딩, 텐서 추론 및 토큰 생성을 비동기로 전담하는 백그라운드 런너입니다.
 * WHY: 대규모 트랜스포머 디코딩 중 메인 스레드 UI 프리징(60fps 끊김)과 브라우저 TDR 락을 원천 차단하기 위함입니다.
 * HOW: postMessage 프로토콜 기반 작업 수신 -> ModelLoader / LLMTextGenerator 실행 -> 실시간 청크 전송.
 */
export interface WorkerInboundMessage {
    type: 'LOAD_MODEL' | 'GENERATE' | 'ABORT';
    id: string;
    payload?: any;
}
export interface WorkerOutboundMessage {
    type: 'LOAD_PROGRESS' | 'LOAD_DONE' | 'TOKEN_STREAM' | 'GENERATE_DONE' | 'ERROR';
    id: string;
    payload?: any;
}
export declare class InferenceWorkerHandler {
    private postMessageFn;
    private session?;
    private abortController?;
    constructor(postMessageFn: (msg: WorkerOutboundMessage) => void);
    handleMessage(msg: WorkerInboundMessage): Promise<void>;
}
