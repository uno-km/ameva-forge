/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: SCRUM-333 GGUF Tensor Mapper Unit Tests
 */

import { GGUFTensorMapper } from '../src/loader/ggufTensorMapper';
import { GGUFStreamer, GGMLType, GGUFHeader } from '../src/loader/ggufStreamer';

describe('GGUFTensorMapper Unit Tests (SCRUM-333)', () => {
  it('accurately converts FP16 half-precision bytes to FP32 float', () => {
    // 0x3C00 is 1.0 in IEEE 754 half-precision
    expect(GGUFTensorMapper.fp16ToFp32(0x3C00)).toBeCloseTo(1.0, 5);
    // 0xBC00 is -1.0
    expect(GGUFTensorMapper.fp16ToFp32(0xBC00)).toBeCloseTo(-1.0, 5);
    // 0x0000 is 0.0
    expect(GGUFTensorMapper.fp16ToFp32(0x0000)).toBe(0.0);
    // 0x4000 is 2.0
    expect(GGUFTensorMapper.fp16ToFp32(0x4000)).toBeCloseTo(2.0, 5);
  });

  it('decodes Q8_0 quantized blocks into Float32Array accurately', () => {
    // 32 elements: scale = 0.5 (0x3800), quants = [1, 2, -1, ...]
    const buffer = new ArrayBuffer(64);
    const view = new DataView(buffer);
    view.setUint16(0, 0x3800, true); // scale = 0.5
    for (let i = 0; i < 32; i++) {
      view.setInt8(2 + i, i % 2 === 0 ? 2 : -2);
    }

    const header: GGUFHeader = {
      magic: 'GGUF',
      version: 3,
      tensorCount: 1,
      metadataKVCount: 0,
      metadata: {},
      tensors: new Map(),
      dataOffset: 0,
    };
    const tensorInfo = {
      name: 'test.weight',
      nDimensions: 1,
      dimensions: [32],
      type: GGMLType.Q8_0,
      offset: 0,
      byteSize: 34,
    };

    const out = GGUFTensorMapper.decodeTensorToFloat32(header, tensorInfo, buffer);
    expect(out.length).toBe(32);
    expect(out[0]).toBeCloseTo(1.0, 4);  // 2 * 0.5 = 1.0
    expect(out[1]).toBeCloseTo(-1.0, 4); // -2 * 0.5 = -1.0
  });

  it('decodes Q4_0 quantized nibble blocks into Float32Array accurately', () => {
    // 32 elements: scale = 1.0 (0x3C00), 16 bytes nibbles
    const buffer = new ArrayBuffer(64);
    const view = new DataView(buffer);
    view.setUint16(0, 0x3C00, true); // scale = 1.0
    // Each byte has low nibble and high nibble: (val - 8) * scale
    // If low nibble is 10 (0x0A), value is 10 - 8 = 2
    // If high nibble is 6 (0x60), value is 6 - 8 = -2
    view.setUint8(2, 0x6A);

    const header: GGUFHeader = {
      magic: 'GGUF',
      version: 3,
      tensorCount: 1,
      metadataKVCount: 0,
      metadata: {},
      tensors: new Map(),
      dataOffset: 0,
    };
    const tensorInfo = {
      name: 'test_q4.weight',
      nDimensions: 1,
      dimensions: [32],
      type: GGMLType.Q4_0,
      offset: 0,
      byteSize: 18,
    };

    const out = GGUFTensorMapper.decodeTensorToFloat32(header, tensorInfo, buffer);
    expect(out[0]).toBeCloseTo(2.0, 4);  // 10 - 8 = 2
    expect(out[1]).toBeCloseTo(-2.0, 4); // 6 - 8 = -2
  });
});
