/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: Classical Computer Vision WebGPU/CPU Kernels
 *
 * WHAT: Sobel 3x3, Canny 에지 검출(8방향 Hysteresis BFS), 가우시안 블러, 그레이스케일 변환을 수행하는 전통 비전 모듈입니다.
 * WHY: VLM 및 딥러닝 추론 전처리, 특징 추출, OCR 사전 처리를 제로 디펜던시로 1ms 내에 완료하기 위해 존재합니다.
 * HOW: 단정밀도 Float32Array 메모리 뷰에서 직접 공간 필터링 및 임계값 추적을 실행합니다.
 */
export declare enum VisionErrorCode {
    INVALID_IMAGE_DIMENSIONS = "INVALID_IMAGE_DIMENSIONS",
    BUFFER_SIZE_MISMATCH = "BUFFER_SIZE_MISMATCH",
    NON_FINITE_PIXEL_VALUE = "NON_FINITE_PIXEL_VALUE",
    THRESHOLD_INVALID = "THRESHOLD_INVALID"
}
export declare class VisionError extends Error {
    readonly code: VisionErrorCode;
    constructor(code: VisionErrorCode, message: string);
}
export declare class ClassicalCV {
    /**
     * RGBA 이미지 버퍼를 단일 채널 그레이스케일 Float32Array[0, 1]로 변환합니다.
     * Y = 0.299*R + 0.587*G + 0.114*B
     */
    static toGrayscale(rgba: Uint8ClampedArray | Uint8Array, width: number, height: number): Float32Array;
    /**
     * 3x3 가우시안 블러 공간 필터링
     */
    static gaussianBlur3x3(input: Float32Array, width: number, height: number): Float32Array;
    /**
     * Sobel 3x3 그래디언트 강도(Magnitude) 및 방향(Angle) 계산
     */
    static sobel3x3(input: Float32Array, width: number, height: number): {
        magnitude: Float32Array;
        angle: Float32Array;
    };
    /**
     * 8-방향 BFS Hysteresis 기반 Canny 에지 검출 알고리즘
     */
    static canny(grayInput: Float32Array, width: number, height: number, lowThreshold?: number, highThreshold?: number): Uint8Array;
}
