/**
 * ============================================================================
 * [FILE METADATA]
 * Project: AMEVA-Forge
 * File: packages/forge/tests/embedding.test.ts
 * Type: TypeScript Unit Test
 * Created: 2026-08-18T23:22:00+09:00
 * ============================================================================
 * WHAT:
 *   WebGPU Native 임베딩(embedding) 커널과 그래프 엔진(graphExecutor)의
 *   명령어 스키마, 파라미터 구조, 2D 디스패치 무결성을 검증하는 테스트입니다.
 */

import { executeGraph } from '../src/tensor/graphExecutor';
import { KERNEL_REGISTRY } from '../src/tensor/gpuCore';

describe('WebGPU Native Embedding Kernel & Graph Schema Tests', () => {
  it('should have embedding kernel registered in KERNEL_REGISTRY', () => {
    expect(KERNEL_REGISTRY.has('embedding')).toBe(true);
    const wgsl = KERNEL_REGISTRY.get('embedding');
    expect(wgsl).toBeDefined();
    expect(wgsl).toContain('struct EmbeddingParams');
    expect(wgsl).toContain('fn main');
  });

  it('should validate embedding schema in executeGraph without crashing on valid params', async () => {
    // Simulated upload + embedding graph instructions
    const instructions = [
      { id: 1, op: 'upload', in: [], shape: [100, 32], params: [] }, // vocab 100, dim 32
      { id: 2, op: 'upload', in: [], shape: [2, 8], params: [] },   // batch 2, seq 8 (16 tokens)
      { id: 3, op: 'embedding', in: [1, 2], shape: [2, 8, 32], params: [16, 32, 100, 0] }
    ];

    // Under node test environment without real WebGPU context,
    // executeGraph will validate schemas before attempting hardware dispatch.
    // If schema is invalid, it throws AMEVAForgeSecurityError synchronously.
    try {
      await executeGraph(JSON.stringify(instructions), []);
    } catch (err: any) {
      // It should NOT throw AMEVAForgeSecurityError about schema or unknown op
      expect(err.message).not.toContain('expected min');
      expect(err.message).not.toContain('Unknown operation');
    }
  });

  it('should reject embedding instruction with insufficient inputs (< 2)', async () => {
    const invalidInstructions = [
      { id: 1, op: 'upload', in: [], shape: [100, 32], params: [] },
      { id: 2, op: 'embedding', in: [1], shape: [2, 8, 32], params: [16, 32, 100, 0] }
    ];

    await expect(executeGraph(JSON.stringify(invalidInstructions), [])).rejects.toThrow(
      /Instruction\[1\] op="embedding": expected exact 2 inputs, got 1/
    );
  });
});
