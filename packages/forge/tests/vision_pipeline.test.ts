/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: Vision Modality Unit Tests (Classical CV, ViT, VLM)
 */

import { ClassicalCV, VisionError, VisionErrorCode } from '../src/vision/classicalCV';
import { CLIPVisionEncoder, CLIPVisionWeights } from '../src/vision/clipVisionEncoder';
import { VLMProjector } from '../src/vision/vlmEngine';

function createDummyLayer(): any {
  const dim = 768;
  return {
    norm1Gamma: new Float32Array(dim).fill(1.0),
    norm1Beta: new Float32Array(dim).fill(0.0),
    qProjWeight: new Float32Array(dim * dim).fill(0.001),
    qProjBias: new Float32Array(dim).fill(0.0),
    kProjWeight: new Float32Array(dim * dim).fill(0.001),
    kProjBias: new Float32Array(dim).fill(0.0),
    vProjWeight: new Float32Array(dim * dim).fill(0.001),
    vProjBias: new Float32Array(dim).fill(0.0),
    outProjWeight: new Float32Array(dim * dim).fill(0.001),
    outProjBias: new Float32Array(dim).fill(0.0),
    norm2Gamma: new Float32Array(dim).fill(1.0),
    norm2Beta: new Float32Array(dim).fill(0.0),
    mlpFc1Weight: new Float32Array(3072 * dim).fill(0.001),
    mlpFc1Bias: new Float32Array(3072).fill(0.0),
    mlpFc2Weight: new Float32Array(dim * 3072).fill(0.001),
    mlpFc2Bias: new Float32Array(dim).fill(0.0),
  };
}

describe('Vision Modality Tests (SCRUM-334)', () => {
  describe('1. Classical CV (Sobel & Canny 8-Direction BFS)', () => {
    it('converts RGBA buffer to grayscale accurately', () => {
      const w = 4, h = 4;
      const rgba = new Uint8ClampedArray(w * h * 4);
      for (let i = 0; i < w * h; i++) {
        rgba[i * 4] = 255;   // R
        rgba[i * 4 + 1] = 0; // G
        rgba[i * 4 + 2] = 0; // B
        rgba[i * 4 + 3] = 255;
      }
      const gray = ClassicalCV.toGrayscale(rgba, w, h);
      expect(gray.length).toBe(16);
      expect(gray[0]).toBeCloseTo(0.299, 3);
    });

    it('detects sharp edge transitions using 8-direction BFS Canny', () => {
      const w = 16, h = 16;
      const gray = new Float32Array(w * h);
      // Create vertical edge in middle
      for (let y = 0; y < h; y++) {
        for (let x = 8; x < w; x++) {
          gray[y * w + x] = 1.0;
        }
      }
      const edges = ClassicalCV.canny(gray, w, h, 0.1, 0.3);
      expect(edges.length).toBe(w * h);
      // Middle column should have edges detected
      let hasEdges = false;
      for (let i = 0; i < edges.length; i++) {
        if (edges[i] === 255) hasEdges = true;
      }
      expect(hasEdges).toBe(true);
    });
  });

  describe('2. CLIP Vision Transformer (ViT-B/16)', () => {
    it('projects 16x16 image patches to 768-dim embeddings', () => {
      const w = 32, h = 32;
      const rgb = new Float32Array(3 * w * h).fill(0.5);
      const weights = new Float32Array(768 * 3 * 16 * 16).fill(0.01);

      const { patches, numPatches } = CLIPVisionEncoder.patchProjection(rgb, w, h, weights);
      expect(numPatches).toBe(4); // (32/16) * (32/16) = 4
      expect(patches.length).toBe(4 * 768);
    });

    it('executes full 1-layer ViT forward pass producing unit-norm image embedding', () => {
      const w = 32, h = 32;
      const rgb = new Float32Array(3 * w * h).fill(0.5);
      const numPatches = 4;

      const classEmbedding = new Float32Array(768);
      for (let i = 0; i < 768; i++) classEmbedding[i] = Math.sin(i * 0.1) * 0.5;

      const weights: CLIPVisionWeights = {
        patchConvWeight: new Float32Array(768 * 3 * 16 * 16).fill(0.01),
        classEmbedding,
        positionEmbedding: new Float32Array((numPatches + 1) * 768).fill(0.01),
        preNormGamma: new Float32Array(768).fill(1.0),
        preNormBeta: new Float32Array(768).fill(0.0),
        layers: [createDummyLayer()],
        postNormGamma: new Float32Array(768).fill(1.0),
        postNormBeta: new Float32Array(768).fill(0.0),
      };

      const { imageEmbedding, patchEmbeddings } = CLIPVisionEncoder.forward(rgb, w, h, weights);
      expect(imageEmbedding.length).toBe(768);
      expect(patchEmbeddings.length).toBe(4 * 768);

      // Verify L2 unit norm
      let normSq = 0;
      for (let i = 0; i < 768; i++) normSq += imageEmbedding[i] * imageEmbedding[i];
      expect(Math.sqrt(normSq)).toBeCloseTo(1.0, 3);
    });
  });

  describe('3. VLM Projector', () => {
    it('projects visual tokens to language model dimension', () => {
      const visualTokens = new Float32Array(4 * 768).fill(0.1);
      const weights = {
        mlp1Weight: new Float32Array(1024 * 768).fill(0.01),
        mlp2Weight: new Float32Array(1024 * 1024).fill(0.01),
      };
      const projected = VLMProjector.project(visualTokens, 4, weights, 1024, 1024);
      expect(projected.length).toBe(4 * 1024);
      for (let i = 0; i < projected.length; i++) {
        expect(Number.isFinite(projected[i])).toBe(true);
      }
    });
  });
});
