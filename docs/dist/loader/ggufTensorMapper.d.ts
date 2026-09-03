/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: SCRUM-333 Real GGUF Model Tensor Mapper & Dequantizer
 *
 * WHAT: GGUF(v2/v3) 바이너리로부터 실제 Stable Diffusion/SDXS 모델의 텐서들을 탐색하고,
 *      FP32, FP16, Q8_0, Q4_0 양자화 가중치를 Float32Array로 역양자화하여
 *      AutoencoderKL, CLIPTextEncoder, UNetGraph의 가중치 구조체로 1:1 바인딩하는 엔진입니다.
 * WHY: 가짜 가중치나 더미 데이터 대신 실제 훈련된 GGUF 체크포인트를 브라우저에서 직접 로드하기 위해 존재합니다.
 * HOW: Half-precision IEEE 754 디코딩, Q8_0/Q4_0 블록 역양자화, 텐서 네이밍 패턴 매칭.
 */
import { GGUFHeader, GGUFTensorInfo } from './ggufStreamer';
import { VAEDecoderWeights } from '../diffusion/vaeDecoder';
export declare class GGUFTensorMapper {
    /**
     * FP16 (IEEE 754 half-precision) 2바이트를 Float32 숫자로 변환합니다.
     */
    static fp16ToFp32(h: number): number;
    /**
     * GGUF 원시 바이너리 버퍼에서 지정된 텐서를 Float32Array로 디코딩합니다.
     */
    static decodeTensorToFloat32(header: GGUFHeader, tensor: GGUFTensorInfo, fileBuffer: ArrayBuffer): Float32Array;
    /**
     * 텐서 이름 검색 (여러 별칭 지원: first_stage_model.*, vae.* 등)
     */
    static findTensor(header: GGUFHeader, patterns: string[]): GGUFTensorInfo | undefined;
    /**
     * GGUF 파일로부터 VAE 디코더 가중치를 추출하여 반환합니다.
     */
    static extractVAEWeights(header: GGUFHeader, fileBuffer: ArrayBuffer): VAEDecoderWeights | undefined;
}
