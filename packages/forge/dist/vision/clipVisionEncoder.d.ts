/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: CLIP ViT-B/16 Vision Transformer Forward Engine
 *
 * WHAT: 이미지를 16x16 패치로 분할하고 트랜스포머 인코더를 거쳐 768차원 시맨틱 특징 벡터를 추출하는 비전 인코더입니다.
 * WHY: 제로샷 이미지 분류, 텍스트-이미지 시맨틱 검색, VLM 멀티모달 시각 입력의 핵심 관문으로 동작합니다.
 * HOW: Patchify Conv2d -> Class Token Concat -> Position Embedding -> 12-Layer Vision Transformer -> LayerNorm.
 */
import { CLIPLayerWeights } from '../diffusion/clipTextEncoder';
export interface CLIPVisionWeights {
    patchConvWeight: Float32Array;
    patchConvBias?: Float32Array;
    classEmbedding: Float32Array;
    positionEmbedding: Float32Array;
    preNormGamma: Float32Array;
    preNormBeta: Float32Array;
    layers: CLIPLayerWeights[];
    postNormGamma: Float32Array;
    postNormBeta: Float32Array;
    projectionWeight?: Float32Array;
}
export declare class CLIPVisionEncoder {
    static readonly PATCH_SIZE = 16;
    static readonly EMBED_DIM = 768;
    static readonly NUM_HEADS = 12;
    /**
     * RGB 이미지(3, H, W)를 16x16 패치로 분할하고 선형 투영합니다.
     */
    static patchProjection(rgb: Float32Array, width: number, height: number, weights: Float32Array, // [768, 3, 16, 16]
    bias?: Float32Array): {
        patches: Float32Array;
        numPatches: number;
    };
    /**
     * CLIP Vision Transformer 전체 순전파:
     * 이미지 RGB -> 패치 임베딩 -> [CLS] 토큰 결합 -> 트랜스포머 레이어 -> 768차원 이미지 특징 벡터
     */
    static forward(rgb: Float32Array, width: number, height: number, weights: CLIPVisionWeights): {
        imageEmbedding: Float32Array;
        patchEmbeddings: Float32Array;
    };
    private static layerNorm;
    private static quickGELU;
    private static linear;
    private static forwardSelfAttention;
}
