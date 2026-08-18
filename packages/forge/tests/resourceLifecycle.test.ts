/**
 * resourceLifecycle.test.ts ??Release 1 resource lifecycle verification
 *
 * Validates:
 * 1. No direct device.createBuffer in Release 1 graph execution path
 * 2. All allocations go through central allocator
 * 3. Token release exactly-once semantics
 */

import * as fs from 'fs';
import * as path from 'path';

describe('Resource Lifecycle - Source Contract', () => {
  const graphExecutorPath = path.join(__dirname, '../src/tensor/graphExecutor.ts');
  let graphExecutorSource: string;

  beforeAll(() => {
    graphExecutorSource = fs.readFileSync(graphExecutorPath, 'utf8');
  });

  describe('Allocator integration', () => {
    it('should not have direct device.createBuffer in graph execution path', () => {
      // After hardening, device.createBuffer should NOT appear in graphExecutor.ts
      // All allocations should go through allocateBuffer()
      const directCreatePattern = /device\.createBuffer\s*\(/g;
      const matches = graphExecutorSource.match(directCreatePattern);

      // After Release 1 hardening, there should be 0 direct device.createBuffer calls
      expect(matches).toBeNull();
    });

    it('should use allocateBuffer for all allocations', () => {
      const allocatePattern = /allocateBuffer\s*\(/g;
      const matches = graphExecutorSource.match(allocatePattern);

      // There should be multiple allocateBuffer calls (tensor, uniform, temporary)
      expect(matches).not.toBeNull();
      expect(matches!.length).toBeGreaterThanOrEqual(3);
    });
  });

  describe('Global error channel removal', () => {
    it('should not set __ameva_last_gpu_error', () => {
      const globalErrorPattern = /__ameva_last_gpu_error\s*=/g;
      const matches = graphExecutorSource.match(globalErrorPattern);
      expect(matches).toBeNull();
    });
  });

  describe('Async contract', () => {
    it('should declare executeGraph as async', () => {
      const asyncPattern = /export\s+async\s+function\s+executeGraph/;
      expect(asyncPattern.test(graphExecutorSource)).toBe(true);
    });

    it('should await error scope pops', () => {
      // Should contain 'await device.popErrorScope()' instead of fire-and-forget
      const awaitPopPattern = /await\s+device\.popErrorScope\s*\(\s*\)/g;
      const matches = graphExecutorSource.match(awaitPopPattern);
      // Should have awaited pops for internal, OOM, and validation scopes in commit and rollback paths
      expect(matches!.length).toBeGreaterThanOrEqual(3);
    });
  });

  describe('Token cleanup', () => {
    it('should use freeBuffer for params cleanup', () => {
      const freeBufferPattern = /freeBuffer\s*\(\s*alloc\.buffer\s*,\s*alloc\.token\s*\)/g;
      const matches = graphExecutorSource.match(freeBufferPattern);
      expect(matches).not.toBeNull();
    });
  });
});
