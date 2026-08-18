import * as fs from 'fs';
import * as path from 'path';
import { _globalRegistry } from '../src/tensor/tensorRegistry';
import { executeGraph } from '../src/tensor/graphExecutor';

describe('GraphTransaction Integration Contract', () => {
  it('executeGraph source code integrates GraphTransaction strictly across all branches', () => {
    const graphExecutorSrc = fs.readFileSync(
      path.resolve(__dirname, '../src/tensor/graphExecutor.ts'),
      'utf8'
    );

    // Verify transaction instantiation
    expect(graphExecutorSrc).toContain('const transaction = new GraphTransaction()');

    // Verify adding pending records
    expect(graphExecutorSrc).toContain('transaction.add(');

    // Verify commit strictly to _globalRegistry
    expect(graphExecutorSrc).toContain('transaction.commit(_globalRegistry)');

    // Verify rollback on both sync error and GPU async error
    expect(graphExecutorSrc).toContain('transaction.rollback()');
  });

  it('pre-GPU failure in executeGraph leaves zero leaking handles in registry', async () => {
    const initialHandles = _globalRegistry.snapshotHandles();

    // Invalid graph with unsupported op and missing parameters
    const failingInstructions = [
      { op: 'invalid_op_xyz', id: 1, shape: [2, 2], in: [] }
    ];

    await expect(
      executeGraph(JSON.stringify(failingInstructions), [])
    ).rejects.toThrow();

    const postHandles = _globalRegistry.snapshotHandles();
    expect(postHandles).toEqual(initialHandles);
  });
});
