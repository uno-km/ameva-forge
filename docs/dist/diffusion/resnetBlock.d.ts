/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-327 UNet ResNet Block WebGPU Forward Pipeline
 *
 * WHAT: 디퓨전 UNet의 기본 연산 단위인 ResNet Block 순전파 오케스트레이터입니다.
 * WHY: GroupNorm, SiLU, Conv2d, Time Embedding Addition, Residual Connection을
 *      WebGPU 상에서 하나의 유기적인 순전파 파이프라인으로 결합하기 위해 존재합니다.
 * HOW: [N, C_in, H, W] 입력에 대해 Norm1 -> SiLU -> Conv1 -> TimeEmbAdd -> Norm2 -> SiLU -> Conv2 -> SkipAdd
 *      연산 그래프를 구성하고 실행합니다.
 */
export interface ResNetBlockWeights {
    norm1Gamma: Float32Array;
    norm1Beta: Float32Array;
    conv1Weight: Float32Array;
    conv1Bias?: Float32Array;
    timeEmbProjWeight?: Float32Array;
    timeEmbProjBias?: Float32Array;
    norm2Gamma: Float32Array;
    norm2Beta: Float32Array;
    conv2Weight: Float32Array;
    conv2Bias?: Float32Array;
    skipProjWeight?: Float32Array;
    skipProjBias?: Float32Array;
}
export interface ResNetBlockConfig {
    inChannels: number;
    outChannels: number;
    height: number;
    width: number;
    numGroups?: number;
}
export declare class ResNetBlock {
    config: ResNetBlockConfig;
    weights: ResNetBlockWeights;
    constructor(config: ResNetBlockConfig, weights: ResNetBlockWeights);
    /**
     * 순수 CPU 참조 수학 연산 (Reference Forward) - WebGPU 출력 결과와의 수치 검증(Numerical Parity)용
     */
    forwardCPU(input: Float32Array, timeEmb?: Float32Array): Float32Array;
    private cpuGroupNorm;
    private cpuSiLU;
    private cpuConv2d;
    private cpuLinear;
}
