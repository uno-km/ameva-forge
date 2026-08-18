/**
 * uniformPool.ts - Transient Uniform Buffer Pool for GraphExecutor & Direct Ops
 * 
 * WHAT: 소형 유니폼 버퍼(Uniform Buffer, 16B~256B)를 고성능으로 재사용하는 전용 버퍼 풀입니다.
 * WHY: 그래프 실행 시 수십 개의 유니폼 버퍼를 매번 allocate/free하면서 onSubmittedWorkDone 지연으로 인해 발생하는 '가짜 OOM(Fake OOM)'을 원천 차단합니다.
 * HOW: 크기별 버킷(16, 32, 64, 112, 144, 256)으로 버퍼를 관리하며, GPU 작업 제출 후 fence 카운터를 통해 안전하게 재사용합니다.
 */
import { getDevice } from "./device";
import { allocateBuffer, freeBuffer } from "./buffers";
import { AllocationToken } from "./quota";

const UNIFORM_BUCKETS = [16, 32, 64, 112, 144, 256, 512, 1024];

export type UniformEntry = {
  buffer: GPUBuffer;
  token: AllocationToken;
  byteLength: number;
  inFlight: boolean;
  fenceId: number;
};

export class UniformBufferPool {
  private pools = new Map<number, UniformEntry[]>();
  private inFlight: UniformEntry[] = [];
  private fenceCounter = 0;

  acquire(byteLength: number): UniformEntry {
    const bucket = this.bucket(byteLength);
    const pool = this.pools.get(bucket) ?? [];

    const reusable = pool.pop();
    if (reusable) {
      reusable.inFlight = true;
      reusable.fenceId = this.fenceCounter;
      return reusable;
    }

    const usage = typeof GPUBufferUsage !== 'undefined'
      ? (GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST)
      : (0x0040 | 0x0008);

    const { buffer, token } = allocateBuffer(
      bucket,
      usage,
      'uniform',
      'UniformBufferPool'
    );

    return {
      buffer,
      token,
      byteLength: bucket,
      inFlight: true,
      fenceId: this.fenceCounter,
    };
  }

  releaseAfterSubmit(entry: UniformEntry): void {
    this.inFlight.push(entry);
  }

  releaseSync(entry: UniformEntry): void {
    entry.inFlight = false;
    try {
      freeBuffer(entry.buffer, entry.token);
    } catch {}
  }

  inFlightBytes(): number {
    return this.inFlight.reduce((acc, e) => acc + e.byteLength, 0);
  }

  async retireSubmitted(device: GPUDevice): Promise<void> {
    const currentFence = ++this.fenceCounter;
    try {
      await device.queue.onSubmittedWorkDone();
    } catch {}

    const stillInFlight: UniformEntry[] = [];
    for (const entry of this.inFlight) {
      if (entry.fenceId < currentFence) {
        entry.inFlight = false;
        const pool = this.pools.get(entry.byteLength) ?? [];
        if (pool.length < 256) {
          pool.push(entry);
          this.pools.set(entry.byteLength, pool);
        } else {
          try { freeBuffer(entry.buffer, entry.token); } catch {}
        }
      } else {
        stillInFlight.push(entry);
      }
    }
    this.inFlight = stillInFlight;
  }

  clear(): void {
    for (const entries of this.pools.values()) {
      for (const entry of entries) {
        try { freeBuffer(entry.buffer, entry.token); } catch {}
      }
    }
    this.pools.clear();
    for (const entry of this.inFlight) {
      try { freeBuffer(entry.buffer, entry.token); } catch {}
    }
    this.inFlight = [];
  }

  private bucket(n: number): number {
    for (const b of UNIFORM_BUCKETS) {
      if (n <= b) return b;
    }
    return Math.ceil(n / 256) * 256;
  }
}

export const _globalUniformPool = new UniformBufferPool();
