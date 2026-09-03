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

export enum GGMLType {
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
  dimensions: number[]; // e.g. [512, 512, 3]
  type: GGMLType;
  offset: number;       // Data section relative offset (bytes)
  byteSize: number;     // Total packed binary size
}

export interface GGUFHeader {
  magic: string;
  version: number;
  tensorCount: number;
  metadataKVCount: number;
  metadata: Record<string, any>;
  tensors: Map<string, GGUFTensorInfo>;
  dataOffset: number;   // Absolute byte offset where tensor data starts
}

export class GGUFStreamer {
  private static readonly GGUF_MAGIC = 0x46554747; // 'GGUF' in LE

  /**
   * 헤더 바이트 버퍼를 파싱하여 메타데이터와 텐서 디스크립터를 추출합니다.
   * 전체 가중치 바이너리가 아닌 헤더 영역(통상 512KB ~ 2MB)만 입력받습니다.
   */
  public static parseHeader(headerBuffer: ArrayBuffer): GGUFHeader {
    const view = new DataView(headerBuffer);
    let offset = 0;

    // 1. Magic 검증
    const magic = view.getUint32(offset, true);
    offset += 4;
    if (magic !== this.GGUF_MAGIC) {
      throw new Error(`[GGUFStreamer] Invalid magic: expected 0x46554747 (GGUF), got 0x${magic.toString(16)}`);
    }

    // 2. Version 검증
    const version = view.getUint32(offset, true);
    offset += 4;
    if (version !== 2 && version !== 3) {
      throw new Error(`[GGUFStreamer] Unsupported GGUF version: ${version} (expected v2 or v3)`);
    }

    // 3. Tensor count & Metadata count (uint64)
    const tensorCount = Number(view.getBigUint64(offset, true));
    offset += 8;
    const metadataKVCount = Number(view.getBigUint64(offset, true));
    offset += 8;

    const metadata: Record<string, any> = {};

    // 4. Metadata KV 파싱
    for (let i = 0; i < metadataKVCount; i++) {
      const keyLen = Number(view.getBigUint64(offset, true));
      offset += 8;
      const keyBytes = new Uint8Array(headerBuffer, offset, keyLen);
      const key = new TextDecoder('utf-8').decode(keyBytes);
      offset += keyLen;

      const valType = view.getUint32(offset, true);
      offset += 4;

      const [val, newOffset] = this.readMetadataValue(view, offset, valType, headerBuffer);
      offset = newOffset;
      metadata[key] = val;
    }

    const alignment = metadata['general.alignment'] ? Number(metadata['general.alignment']) : 32;

    // 5. Tensors Table 파싱
    const tensors = new Map<string, GGUFTensorInfo>();
    for (let i = 0; i < tensorCount; i++) {
      const nameLen = Number(view.getBigUint64(offset, true));
      offset += 8;
      const nameBytes = new Uint8Array(headerBuffer, offset, nameLen);
      const name = new TextDecoder('utf-8').decode(nameBytes);
      offset += nameLen;

      const nDims = view.getUint32(offset, true);
      offset += 4;

      const dimensions: number[] = [];
      let totalElements = 1;
      for (let d = 0; d < nDims; d++) {
        const dim = Number(view.getBigUint64(offset, true));
        offset += 8;
        dimensions.push(dim);
        totalElements *= dim;
      }

      const type = view.getUint32(offset, true) as GGMLType;
      offset += 4;

      const tensorOffset = Number(view.getBigUint64(offset, true));
      offset += 8;

      const byteSize = this.calculateTensorByteSize(type, totalElements);

      tensors.set(name, {
        name,
        nDimensions: nDims,
        dimensions,
        type,
        offset: tensorOffset,
        byteSize,
      });
    }

    // 6. Data 섹션 시작 주소 정렬 (alignment padding)
    const remainder = offset % alignment;
    const dataOffset = remainder === 0 ? offset : offset + (alignment - remainder);

    return {
      magic: 'GGUF',
      version,
      tensorCount,
      metadataKVCount,
      metadata,
      tensors,
      dataOffset,
    };
  }

  /**
   * 개별 텐서 바이너리를 수신하여 WASM 힙을 거치지 않고 WebGPU GPUBuffer로 직분사(Direct Injection)합니다.
   */
  public static async injectTensorToWebGPU(
    device: GPUDevice,
    tensorInfo: GGUFTensorInfo,
    chunkFetcher: () => Promise<ArrayBuffer>,
    usage: GPUBufferUsageFlags = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  ): Promise<GPUBuffer> {
    // 1. VRAM 버퍼 할당 (4바이트 배수 올림)
    const alignedSize = Math.ceil(tensorInfo.byteSize / 4) * 4;
    const gpuBuffer = device.createBuffer({
      label: `gguf_tensor_${tensorInfo.name}`,
      size: alignedSize,
      usage,
    });

    // 2. 외부 청크 페치 (네트워크 스트림 or OPFS)
    const binaryChunk = await chunkFetcher();
    if (binaryChunk.byteLength < tensorInfo.byteSize) {
      throw new Error(
        `[GGUFStreamer] Chunk size mismatch for tensor ${tensorInfo.name}: expected ${tensorInfo.byteSize} bytes, got ${binaryChunk.byteLength}`
      );
    }

    // 3. WebGPU 큐를 통한 비동기 직분사 DMA (Zero WASM Heap)
    device.queue.writeBuffer(
      gpuBuffer,
      0,
      binaryChunk,
      0,
      tensorInfo.byteSize
    );

    return gpuBuffer;
  }

  private static calculateTensorByteSize(type: GGMLType, numElements: number): number {
    switch (type) {
      case GGMLType.F32:
        return numElements * 4;
      case GGMLType.F16:
        return numElements * 2;
      case GGMLType.Q8_0: {
        // block size 32: 32 elements = 2 bytes scale (fp16) + 32 bytes quants = 34 bytes
        const blocks = Math.ceil(numElements / 32);
        return blocks * 34;
      }
      case GGMLType.Q4_0: {
        // block size 32: 32 elements = 2 bytes scale (fp16) + 16 bytes quants (4bit) = 18 bytes
        const blocks = Math.ceil(numElements / 32);
        return blocks * 18;
      }
      case GGMLType.Q4_K: {
        // block size 256: 256 elements = 144 bytes
        const blocks = Math.ceil(numElements / 256);
        return blocks * 144;
      }
      default:
        // 폴백 기본 계산 (안전 마진)
        return numElements * 4;
    }
  }

  private static readMetadataValue(
    view: DataView,
    offset: number,
    valType: number,
    buffer: ArrayBuffer
  ): [any, number] {
    switch (valType) {
      case 0: // UINT8
        return [view.getUint8(offset), offset + 1];
      case 1: // INT8
        return [view.getInt8(offset), offset + 1];
      case 2: // UINT16
        return [view.getUint16(offset, true), offset + 2];
      case 3: // INT16
        return [view.getInt16(offset, true), offset + 2];
      case 4: // UINT32
        return [view.getUint32(offset, true), offset + 4];
      case 5: // INT32
        return [view.getInt32(offset, true), offset + 4];
      case 6: // FLOAT32
        return [view.getFloat32(offset, true), offset + 4];
      case 7: // BOOL
        return [view.getUint8(offset) !== 0, offset + 1];
      case 8: { // STRING
        const strLen = Number(view.getBigUint64(offset, true));
        offset += 8;
        const strBytes = new Uint8Array(buffer, offset, strLen);
        const str = new TextDecoder('utf-8').decode(strBytes);
        return [str, offset + strLen];
      }
      case 9: { // ARRAY
        const itemType = view.getUint32(offset, true);
        offset += 4;
        const arrLen = Number(view.getBigUint64(offset, true));
        offset += 8;
        const arr: any[] = [];
        for (let i = 0; i < arrLen; i++) {
          const [elem, nextOffset] = this.readMetadataValue(view, offset, itemType, buffer);
          arr.push(elem);
          offset = nextOffset;
        }
        return [arr, offset];
      }
      case 10: // UINT64
        return [view.getBigUint64(offset, true), offset + 8];
      case 11: // INT64
        return [view.getBigInt64(offset, true), offset + 8];
      case 12: // FLOAT64
        return [view.getFloat64(offset, true), offset + 8];
      default:
        throw new Error(`[GGUFStreamer] Unknown metadata value type: ${valType}`);
    }
  }
}
