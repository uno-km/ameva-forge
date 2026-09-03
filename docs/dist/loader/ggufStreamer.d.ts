/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-318 Zero-Heap GGUF Header & Tensor Streaming Parser
 *
 * WHAT: GGUF(v2/v3) 바이너리 모델 파일의 헤더 및 텐서 메타데이터를 파싱하고,
 *      가중치 데이터를 WASM 힙 메모리를 우회하여 WebGPU VRAM 버퍼로 직접 주입(Direct DMA)하는 고성능 스트리머입니다.
 * WHY: 32비트 WebAssembly(WASM) 환경의 2GB 힙 한계로 인한 브라우저 OOM 크래시를 원천 차단하고,
 *      1.5GB 이상의 Stable Diffusion GGUF 가중치를 Zero-Heap으로 VRAM에 안전하게 적재하기 위해 존재합니다.
 * HOW: 최초 1~2MB 헤더 블록만 읽어 메타데이터 딕셔너리와 텐서 테이블을 구축한 후,
 *      HTTP Range-Request 또는 OPFS 스트림을 통해 필요한 텐서 청크만 직접 WebGPU Queue.writeBuffer로 전송합니다.
 */
export declare enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
    COUNT = 19
}
export interface GGUFTensorInfo {
    name: string;
    nDimensions: number;
    dimensions: number[];
    type: GGMLType;
    offset: number;
    byteSize: number;
}
export interface GGUFHeader {
    magic: string;
    version: number;
    tensorCount: number;
    metadataKVCount: number;
    metadata: Record<string, any>;
    tensors: Map<string, GGUFTensorInfo>;
    dataOffset: number;
}
export declare class GGUFStreamer {
    private static readonly GGUF_MAGIC;
    /**
     * 헤더 바이트 버퍼를 파싱하여 메타데이터와 텐서 디스크립터를 추출합니다.
     * 전체 가중치 바이너리가 아닌 헤더 영역(통상 512KB ~ 2MB)만 입력받습니다.
     */
    static parseHeader(headerBuffer: ArrayBuffer): GGUFHeader;
    /**
     * 개별 텐서 바이너리를 수신하여 WASM 힙을 거치지 않고 WebGPU GPUBuffer로 직분사(Direct Injection)합니다.
     */
    static injectTensorToWebGPU(device: GPUDevice, tensorInfo: GGUFTensorInfo, chunkFetcher: () => Promise<ArrayBuffer>, usage?: GPUBufferUsageFlags): Promise<GPUBuffer>;
    private static calculateTensorByteSize;
    private static readMetadataValue;
}
