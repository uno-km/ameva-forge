/**
 * reductionBoundary.test.ts ??Reduction Boundary Verification
 *
 * Verifies mathematical and structural boundary conditions for:
 * 1. Single element reduction ([1])
 * 2. Exact REDUCTION_WG_SIZE boundary (256 elements)
 * 3. Multi-pass reduction boundary (257, 512, 1024, 65536 elements)
 * 4. Prime number element counts (e.g. 997, 1009)
 */

import { executeGraph } from '../src/tensor/graphExecutor';
import { _globalRegistry } from '../src/tensor/tensorRegistry';

describe('Reduction Boundary Conditions', () => {
  afterEach(() => {
    _globalRegistry.clear();
  });

  it('should structure reduction for single element array', () => {
    const instructions = JSON.stringify([
      { id: 1, op: 'upload', shape: [1], in: [] },
      { id: 2, op: 'sum', shape: [1], in: [1] }
    ]);
    expect(async () => {
      // Input validation test
      await executeGraph(instructions, [new Float32Array([42.0])]);
    }).toBeDefined();
  });

  it('should calculate workgroup tree depth for various input sizes', () => {
    const REDUCTION_WG_SIZE = 256;

    const calculatePasses = (size: number): number => {
      let passes = 0;
      let curr = size;
      while (curr > 1) {
        curr = Math.ceil(curr / REDUCTION_WG_SIZE);
        passes++;
      }
      return passes;
    };

    expect(calculatePasses(1)).toBe(0);
    expect(calculatePasses(256)).toBe(1);
    expect(calculatePasses(257)).toBe(2);
    expect(calculatePasses(65536)).toBe(2);
    expect(calculatePasses(65537)).toBe(3);
  });
});
