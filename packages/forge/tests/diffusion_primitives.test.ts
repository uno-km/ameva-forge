/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-310 ~ SCRUM-318 Diffusion Primitives & Zero-Heap GGUF Streamer Tests
 */

import { GGUFStreamer, GGMLType } from '../src/loader/ggufStreamer';
import { SILU_WGSL, SILU_BACKWARD_WGSL } from '../src/tensor/kernels/silu.wgsl';
import { UPSAMPLE2D_WGSL } from '../src/tensor/kernels/upsample2d.wgsl';
import { GROUP_NORM_STATS_WGSL, GROUP_NORM_APPLY_WGSL } from '../src/tensor/kernels/group_norm.wgsl';

describe('Diffusion WebGPU Primitives & Zero-Heap GGUF Streamer (SCRUM-310 ~ SCRUM-318)', () => {
  describe('1. WGSL Compute Shaders Integrity', () => {
    it('SiLU forward & backward shaders declare valid uniforms and workgroup entrypoints', () => {
      expect(SILU_WGSL).toContain('@compute @workgroup_size(64, 1, 1)');
      expect(SILU_WGSL).toContain('struct Params');
      expect(SILU_WGSL).toContain('stable_sigmoid');
      expect(SILU_BACKWARD_WGSL).toContain('@compute @workgroup_size(64, 1, 1)');
      expect(SILU_BACKWARD_WGSL).toContain('grad_input[idx] = grad_output[idx]');
    });

    it('Upsample2D shader supports Nearest and Bilinear modes with 16-byte aligned Params', () => {
      expect(UPSAMPLE2D_WGSL).toContain('@compute @workgroup_size(64, 1, 1)');
      expect(UPSAMPLE2D_WGSL).toContain('mode == 0u'); // Nearest
      expect(UPSAMPLE2D_WGSL).toContain('real_h'); // Bilinear
      expect(UPSAMPLE2D_WGSL).toContain('struct Params');
    });

    it('GroupNorm 2-Pass shaders declare workgroup shared memory reduction and Affine/SiLU fusion', () => {
      expect(GROUP_NORM_STATS_WGSL).toContain('var<workgroup> wg_sum: array<f32, 64>;');
      expect(GROUP_NORM_STATS_WGSL).toContain('workgroupBarrier();');
      expect(GROUP_NORM_APPLY_WGSL).toContain('inverseSqrt(variance + params.eps)');
      expect(GROUP_NORM_APPLY_WGSL).toContain('fuse_silu == 1u');
    });
  });

  describe('2. Zero-Heap GGUF Binary Streamer', () => {
    it('successfully parses a synthetic GGUF v3 header without touching WASM memory', () => {
      // Build a minimal valid GGUF v3 buffer in memory
      const buffer = new ArrayBuffer(4096);
      const view = new DataView(buffer);
      const u8 = new Uint8Array(buffer);
      let offset = 0;

      // Magic: 'GGUF' (0x46554747 in LE)
      view.setUint32(offset, 0x46554747, true);
      offset += 4;

      // Version: 3
      view.setUint32(offset, 3, true);
      offset += 4;

      // Tensor Count: 1
      view.setBigUint64(offset, 1n, true);
      offset += 8;

      // Metadata Count: 1
      view.setBigUint64(offset, 1n, true);
      offset += 8;

      // Key: 'general.architecture' (len = 20)
      const keyStr = 'general.architecture';
      view.setBigUint64(offset, BigInt(keyStr.length), true);
      offset += 8;
      for (let i = 0; i < keyStr.length; i++) {
        u8[offset + i] = keyStr.charCodeAt(i);
      }
      offset += keyStr.length;

      // Value type: STRING (8)
      view.setUint32(offset, 8, true);
      offset += 4;

      // Value: 'diffusion' (len = 9)
      const valStr = 'diffusion';
      view.setBigUint64(offset, BigInt(valStr.length), true);
      offset += 8;
      for (let i = 0; i < valStr.length; i++) {
        u8[offset + i] = valStr.charCodeAt(i);
      }
      offset += valStr.length;

      // Tensor 0: 'unet.conv_in.weight'
      const tensorName = 'unet.conv_in.weight';
      view.setBigUint64(offset, BigInt(tensorName.length), true);
      offset += 8;
      for (let i = 0; i < tensorName.length; i++) {
        u8[offset + i] = tensorName.charCodeAt(i);
      }
      offset += tensorName.length;

      // nDims = 4
      view.setUint32(offset, 4, true);
      offset += 4;

      // Dims: [320, 4, 3, 3] = 11,520 elements
      view.setBigUint64(offset, 320n, true);
      offset += 8;
      view.setBigUint64(offset, 4n, true);
      offset += 8;
      view.setBigUint64(offset, 3n, true);
      offset += 8;
      view.setBigUint64(offset, 3n, true);
      offset += 8;

      // Type: Q4_K (12)
      view.setUint32(offset, GGMLType.Q4_K, true);
      offset += 4;

      // Offset: 0
      view.setBigUint64(offset, 0n, true);
      offset += 8;

      // Execute parseHeader
      const header = GGUFStreamer.parseHeader(buffer.slice(0, offset));

      expect(header.magic).toBe('GGUF');
      expect(header.version).toBe(3);
      expect(header.tensorCount).toBe(1);
      expect(header.metadata['general.architecture']).toBe('diffusion');

      const tensorInfo = header.tensors.get('unet.conv_in.weight');
      expect(tensorInfo).toBeDefined();
      expect(tensorInfo?.dimensions).toEqual([320, 4, 3, 3]);
      expect(tensorInfo?.type).toBe(GGMLType.Q4_K);
      // 11520 elements in Q4_K (blocks of 256): ceil(11520 / 256) * 144 = 45 * 144 = 6480 bytes
      expect(tensorInfo?.byteSize).toBe(45 * 144);
    });

    it('rejects invalid GGUF magic with descriptive error', () => {
      const badBuffer = new ArrayBuffer(64);
      const view = new DataView(badBuffer);
      view.setUint32(0, 0x12345678, true);

      expect(() => GGUFStreamer.parseHeader(badBuffer)).toThrow(
        '[GGUFStreamer] Invalid magic: expected 0x46554747 (GGUF)'
      );
    });
  });
});
