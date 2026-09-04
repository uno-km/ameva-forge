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

import { WorkerInboundMessage, WorkerOutboundMessage } from './inferenceWorker';
import { ModelLoadProgress } from '../loader/modelLoader';
import { TextGenerationProgress } from '../llm/llmTextGenerator';

export interface WorkerSessionOptions {
  workerUrl?: string;
  onProgress?: (progress: ModelLoadProgress) => void;
  useCache?: boolean;
}

export class WorkerSession {
  private worker: Worker;
  private reqSeq: number = 0;
  private pendingResolvers: Map<string, { resolve: (val: any) => void; reject: (err: any) => void }> = new Map();
  private streamHandlers: Map<string, (token: string, progress: TextGenerationProgress) => void> = new Map();
  private progressHandler?: (progress: ModelLoadProgress) => void;

  constructor(worker: Worker, onProgress?: (p: ModelLoadProgress) => void) {
    this.worker = worker;
    this.progressHandler = onProgress;
    this.worker.onmessage = this.handleWorkerMessage.bind(this);
  }

  private handleWorkerMessage(e: MessageEvent<WorkerOutboundMessage>): void {
    const { type, id, payload } = e.data;

    switch (type) {
      case 'LOAD_PROGRESS':
        this.progressHandler?.(payload);
        break;

      case 'LOAD_DONE': {
        const resolver = this.pendingResolvers.get(id);
        if (resolver) {
          resolver.resolve(payload);
          this.pendingResolvers.delete(id);
        }
        break;
      }

      case 'TOKEN_STREAM': {
        const handler = this.streamHandlers.get(id);
        if (handler) {
          handler(payload.token, payload.progress);
        }
        break;
      }

      case 'GENERATE_DONE': {
        const resolver = this.pendingResolvers.get(id);
        if (resolver) {
          resolver.resolve(payload.fullText);
          this.pendingResolvers.delete(id);
          this.streamHandlers.delete(id);
        }
        break;
      }

      case 'ERROR': {
        const resolver = this.pendingResolvers.get(id);
        if (resolver) {
          resolver.reject(new Error(payload.message));
          this.pendingResolvers.delete(id);
          this.streamHandlers.delete(id);
        }
        break;
      }
    }
  }

  /**
   * 백그라운드 Worker에 자연어 텍스트 프롬프트를 전송하고 스트리밍으로 결과를 수신합니다.
   */
  public async prompt(
    text: string,
    options: {
      maxNewTokens?: number;
      temperature?: number;
      topK?: number;
      topP?: number;
      backend?: 'webgpu' | 'cpu';
      onToken?: (token: string, progress: TextGenerationProgress) => void;
    } = {}
  ): Promise<string> {
    const id = `req_${++this.reqSeq}_${Date.now()}`;

    if (options.onToken) {
      this.streamHandlers.set(id, options.onToken);
    }

    return new Promise((resolve, reject) => {
      this.pendingResolvers.set(id, { resolve, reject });

      this.worker.postMessage({
        type: 'GENERATE',
        id,
        payload: {
          prompt: text,
          maxNewTokens: options.maxNewTokens,
          temperature: options.temperature,
          topK: options.topK,
          topP: options.topP,
          backend: options.backend,
        },
      } as WorkerInboundMessage);
    });
  }

  /**
   * 실행 중인 텍스트 생성을 즉시 중단합니다.
   */
  public abort(): void {
    this.worker.postMessage({
      type: 'ABORT',
      id: 'abort_current',
    } as WorkerInboundMessage);
  }

  /**
   * Worker 스레드를 영구 종료하고 자원을 해제합니다.
   */
  public terminate(): void {
    this.worker.terminate();
    this.pendingResolvers.clear();
    this.streamHandlers.clear();
  }
}
