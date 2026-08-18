import { _stagingPool, acquireStagingBuffer, releaseStagingBuffer, clearStagingPool } from '../src/webgpu/buffers';
import { _globalUniformPool } from '../src/webgpu/uniformPool';
import { _resetDeviceForTesting } from '../src/webgpu/device';

describe('Staging & Uniform Buffer Pooling Resilience', () => {
  beforeEach(() => {
    clearStagingPool();
    _globalUniformPool.clear();
  });

  it('reuses staging buffers within pool limit instead of creating new GPUBuffer instances', () => {
    const byteLength = 1024;
    const entry1 = acquireStagingBuffer(byteLength);
    expect(entry1.buffer).toBeDefined();

    // Release back to pool
    releaseStagingBuffer(entry1.buffer, entry1.token, byteLength, false);
    expect(_stagingPool.get(byteLength)?.length).toBe(1);

    // Acquire again - must return the same buffer
    const entry2 = acquireStagingBuffer(byteLength);
    expect(entry2.buffer).toBe(entry1.buffer);
    expect(entry2.token).toBe(entry1.token);
    expect(_stagingPool.get(byteLength)?.length).toBe(0);

    releaseStagingBuffer(entry2.buffer, entry2.token, byteLength, false);
  });

  it('UniformBufferPool reuses buffers after submission retirement', async () => {
    const u1 = _globalUniformPool.acquire(112);
    expect(u1.buffer).toBeDefined();
    expect(u1.byteLength).toBe(112);

    _globalUniformPool.releaseAfterSubmit(u1);
    expect(_globalUniformPool.inFlightBytes()).toBe(112);

    // Simulate work done
    const mockDevice = {
      queue: {
        onSubmittedWorkDone: jest.fn().mockResolvedValue(undefined),
      },
    } as any;

    await _globalUniformPool.retireSubmitted(mockDevice);
    expect(_globalUniformPool.inFlightBytes()).toBe(0);

    // Next acquire gets the pooled uniform entry
    const u2 = _globalUniformPool.acquire(112);
    expect(u2.buffer).toBe(u1.buffer);
  });
});
