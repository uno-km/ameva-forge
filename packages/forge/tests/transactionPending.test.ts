import { TensorRegistry } from '../src/tensor/tensorRegistry';
import { GraphTransaction, PendingTensorRecord } from '../src/tensor/graphExecutor';
import { AllocationToken } from '../src/webgpu/quota';

describe('GraphTransaction pending and committed isolation', () => {
  it('pending handles are not visible in registry until committed', () => {
    const registry = new TensorRegistry();
    const transaction = new GraphTransaction();

    const mockBuffer = { destroy: jest.fn() } as unknown as GPUBuffer;
    const mockToken = new AllocationToken('alloc_1', 16, 'tensor', 'g1', 1);

    const pendingRecord: PendingTensorRecord = {
      handle: 'tensor_test_123',
      buffer: mockBuffer,
      token: mockToken,
      shape: [2, 2],
      dtype: 'float32',
      byteLength: 16,
    };

    transaction.add(pendingRecord);

    // Not in registry before commit
    expect(registry.has('tensor_test_123')).toBe(false);
    expect(registry.snapshotHandles()).not.toContain('tensor_test_123');

    // Commit
    transaction.commit(registry);

    // Visible in registry after commit
    expect(registry.has('tensor_test_123')).toBe(true);
    expect(registry.snapshotHandles()).toContain('tensor_test_123');
  });

  it('rollback destroys pending buffer without touching registry', () => {
    const registry = new TensorRegistry();
    const transaction = new GraphTransaction();

    const destroyMock = jest.fn();
    const mockBuffer = { destroy: destroyMock } as unknown as GPUBuffer;
    const mockToken = new AllocationToken('alloc_2', 32, 'tensor', 'g2', 1);

    const pendingRecord: PendingTensorRecord = {
      handle: 'tensor_test_rollback',
      buffer: mockBuffer,
      token: mockToken,
      shape: [4, 2],
      dtype: 'float32',
      byteLength: 32,
    };

    transaction.add(pendingRecord);
    transaction.rollback();

    expect(destroyMock).toHaveBeenCalledTimes(1);
    expect(registry.has('tensor_test_rollback')).toBe(false);
    expect(registry.snapshotHandles()).toEqual([]);
  });
});
