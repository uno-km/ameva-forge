import { QuotaManager } from "../src/webgpu/quota";
import { AMEVAForgeQuotaExceededError } from "../src/errors";
import { _globalRegistry } from "../src/tensor/tensorRegistry";

describe("Stress and Security Test (F-039)", () => {
  beforeEach(() => {
    _globalRegistry.clear();
  });

  it("handles boundary values correctly", () => {
    const q = new QuotaManager(1024, 1024);

    // Exactly at limit should pass
    const t1 = q.reserveToken(1024, 'tensor', '');
    expect(q.getUsage().allocatedBytes).toBe(1024);
    q.releaseToken(t1);

    // One byte over limit should fail
    expect(() => q.reserveToken(1025, 'tensor', '')).toThrow(AMEVAForgeQuotaExceededError);
  });

  it("prevents async race conditions on allocation", async () => {
    const q = new QuotaManager(1000, 1000);

    // Attempt 100 concurrent allocations of 10 bytes each
    const promises = Array.from({ length: 100 }, async (_, i) => {
      // Simulate async delay
      await new Promise(r => setTimeout(r, Math.random() * 5));
      return q.reserveToken(10, 'tensor', `id_${i}`);
    });

    const tokens = await Promise.all(promises);
    expect(q.getUsage().allocatedBytes).toBe(1000);

    // Any further allocation should fail
    expect(() => q.reserveToken(1, 'tensor', 'over')).toThrow(AMEVAForgeQuotaExceededError);

    // Release all
    tokens.forEach(t => q.releaseToken(t));
    expect(q.getUsage().allocatedBytes).toBe(0);
  });

  it("defends against negative allocations", () => {
    const q = new QuotaManager(1000, 1000);
    expect(() => q.reserveToken(-500, 'tensor', '')).toThrow(AMEVAForgeQuotaExceededError);
    // Usage should remain 0
    expect(q.getUsage().allocatedBytes).toBe(0);
  });

  it("OOM safety when allocating extremely large tensors", () => {
    const q = new QuotaManager(1000000, 1000000); // 1MB limit

    // Allocate 2GB (exceeds limit)
    expect(() => q.reserveToken(2 * 1024 * 1024 * 1024, 'tensor', '')).toThrow(AMEVAForgeQuotaExceededError);
  });
});
