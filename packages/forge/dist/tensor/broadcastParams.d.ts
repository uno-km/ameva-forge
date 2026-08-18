/**
 * 파일 생성일: 2026-08-18T14:42:00+09:00
 * 역할: 8D 다차원 스트라이드 브로드캐스팅 파라미터 계산 유틸리티
 * 목적: Direct TS API(gpuCore.ts)와 Graph Executor(graphExecutor.ts) 간의 112-Byte WGSL 유니폼 버퍼 계약 일치
 */
export interface BroadcastParams {
    dOut: number[];
    effSA: number[];
    effSB: number[];
}
/**
 * WHAT: 두 텐서의 형태(shapeA, shapeB)를 8차원으로 좌측 패딩(pad8)하고 유효 스트라이드를 계산합니다.
 * WHY: WGSL 셰이더(ADD, SUB, MUL, DIV)가 8차원 좌표 디코딩을 수행할 때 정확한 메모리 오프셋을 역산할 수 있도록 하기 위함입니다.
 * HOW: 8차원으로 정규화 후 역순 스트라이드를 계산하고, 크기가 1인 차원은 스트라이드를 0으로 매핑(브로드캐스팅)합니다.
 */
export declare function computeBroadcastParams(outShape: number[], shapeA: number[], shapeB: number[]): BroadcastParams;
