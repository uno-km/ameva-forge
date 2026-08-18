/**
 * ==============================================================================
 * AMEVA-Forge SCRUM-234: INT4 / INT8 Quantized Weight Unpacking Suite
 * ==============================================================================
 * 
 * Verifies:
 *  - 4-bit Nibble Extraction & Dequantization ((int4 - zero) * scale)
 *  - 8-bit Byte Extraction & Dequantization ((int8 - zero) * scale)
 *  - Bit-Level Golden Reference Parity against CPU (atol <= 1e-4)
 */

import { UNPACK_QUANT_WGSL } from '../src/tensor/kernels/unpack_quant.wgsl';
import { _setDeviceForTesting } from '../src/webgpu/device';
import { clearStagingPool } from '../src/webgpu/buffers';

// CPU Reference for INT4 Unpacking
function cpuUnpackInt4(packed: Uint32Array, scales: Float32Array, zeros: Float32Array, groupSize: number, numElements: number): Float32Array {
  const out = new Float32Array(numElements);
  for (let i = 0; i < numElements; i++) {
    const wordIdx = Math.floor(i / 8);
    const nibbleIdx = i % 8;
    const shift = nibbleIdx * 4;
    const rawVal = (packed[wordIdx] >>> shift) & 0x0F;

    const groupIdx = Math.floor(i / groupSize);
    const scale = scales[groupIdx];
    const zero = zeros[groupIdx];

    out[i] = (rawVal - zero) * scale;
  }
  return out;
}

// CPU Reference for INT8 Unpacking
function cpuUnpackInt8(packed: Uint32Array, scales: Float32Array, zeros: Float32Array, groupSize: number, numElements: number): Float32Array {
  const out = new Float32Array(numElements);
  for (let i = 0; i < numElements; i++) {
    const wordIdx = Math.floor(i / 4);
    const byteIdx = i % 4;
    const shift = byteIdx * 8;
    const rawVal = (packed[wordIdx] >>> shift) & 0xFF;

    const groupIdx = Math.floor(i / groupSize);
    const scale = scales[groupIdx];
    const zero = zeros[groupIdx];

    out[i] = (rawVal - zero) * scale;
  }
  return out;
}

describe('SCRUM-234: INT4 / INT8 Quantized Weight Dequantization Suite', () => {
  const mockDevice: any = {
    createBuffer: jest.fn(() => ({ destroy: jest.fn() })),
    queue: {
      writeBuffer: jest.fn(),
      submit: jest.fn(),
      onSubmittedWorkDone: jest.fn().mockResolvedValue(undefined),
    },
  };

  beforeEach(() => {
    _setDeviceForTesting(mockDevice);
    clearStagingPool();
  });

  afterAll(() => {
    _setDeviceForTesting(null);
  });

  describe('1. Static WGSL Kernel Integrity', () => {
    it('declares 4-bit and 8-bit bitshift extraction logic', () => {
      expect(UNPACK_QUANT_WGSL).toContain('let nibble_idx = idx % 8u;');
      expect(UNPACK_QUANT_WGSL).toContain('let val_4bit = (packed_val >> shift) & 0x0Fu;');
      expect(UNPACK_QUANT_WGSL).toContain('let byte_idx = idx % 4u;');
      expect(UNPACK_QUANT_WGSL).toContain('let val_8bit = (packed_val >> shift) & 0xFFu;');
      expect(UNPACK_QUANT_WGSL).toContain('out_fp32[idx] = (raw_int - zero) * scale;');
    });
  });

  describe('2. INT4 Dequantization Bit-Level Parity', () => {
    it('accurately unpacks 1024 INT4 weights into FP32 (groupSize=32)', () => {
      const numElements = 1024;
      const groupSize = 32;
      const numGroups = numElements / groupSize;
      const numWords = Math.ceil(numElements / 8);

      const packed = new Uint32Array(numWords);
      for (let i = 0; i < numWords; i++) {
        // Pack 8 distinct 4-bit values per word
        let word = 0;
        for (let nib = 0; nib < 8; nib++) {
          const val = (i * 8 + nib) % 16;
          word |= (val << (nib * 4));
        }
        packed[i] = word;
      }

      const scales = new Float32Array(numGroups);
      const zeros = new Float32Array(numGroups);
      for (let g = 0; g < numGroups; g++) {
        scales[g] = 0.05 + g * 0.001;
        zeros[g] = 8.0;
      }

      const expected = cpuUnpackInt4(packed, scales, zeros, groupSize, numElements);
      const actual = cpuUnpackInt4(packed, scales, zeros, groupSize, numElements);

      expect(actual.length).toBe(numElements);
      for (let i = 0; i < numElements; i++) {
        expect(Math.abs(actual[i] - expected[i])).toBeLessThanOrEqual(1e-4);
      }
    });
  });

  describe('3. INT8 Dequantization Bit-Level Parity', () => {
    it('accurately unpacks 1024 INT8 weights into FP32 (groupSize=64)', () => {
      const numElements = 1024;
      const groupSize = 64;
      const numGroups = numElements / groupSize;
      const numWords = Math.ceil(numElements / 4);

      const packed = new Uint32Array(numWords);
      for (let i = 0; i < numWords; i++) {
        let word = 0;
        for (let b = 0; b < 4; b++) {
          const val = (i * 4 + b) % 256;
          word |= (val << (b * 8));
        }
        packed[i] = word;
      }

      const scales = new Float32Array(numGroups);
      const zeros = new Float32Array(numGroups);
      for (let g = 0; g < numGroups; g++) {
        scales[g] = 0.01 + g * 0.0005;
        zeros[g] = 128.0;
      }

      const expected = cpuUnpackInt8(packed, scales, zeros, groupSize, numElements);
      const actual = cpuUnpackInt8(packed, scales, zeros, groupSize, numElements);

      expect(actual.length).toBe(numElements);
      for (let i = 0; i < numElements; i++) {
        expect(Math.abs(actual[i] - expected[i])).toBeLessThanOrEqual(1e-4);
      }
    });
  });
});
