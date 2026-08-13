import { QuotaManager } from "../src/webgpu/quota";
import { AMEVAForgeQuotaExceededError } from "../src/errors";

describe("QuotaManager", () => {
  it("tracks quota correctly", () => {
    const q = new QuotaManager(1000, 800);
    const t1 = q.reserveToken(300, 'tensor', '');
    expect(q.getUsage().allocatedBytes).toBe(300);
    q.releaseToken(t1);
    expect(q.getUsage().allocatedBytes).toBe(0);
  });

  it("throws error when exceeding limits", () => {
    const q = new QuotaManager(1000, 800);
    expect(() => q.reserveToken(1001, 'tensor', '')).toThrow(AMEVAForgeQuotaExceededError);
  });

  it("rejects invalid allocation sizes", () => {
    const q = new QuotaManager(1000, 800);
    expect(() => q.reserveToken(-100, 'tensor', '')).toThrow(AMEVAForgeQuotaExceededError);
    expect(() => q.reserveToken(10.5, 'tensor', '')).toThrow(AMEVAForgeQuotaExceededError);
  });

  it("prevents underflow", () => {
    const q = new QuotaManager(1000, 800);
    const t1 = q.reserveToken(300, 'tensor', '');
    q.releaseToken(t1);
    expect(() => q.releaseToken(t1)).not.toThrow();
    expect(q.getUsage().allocatedBytes).toBe(0); // Should still be 0 if double released (not possible but safe)
  });

  it("warns when exceeding soft limit", () => {
    const q = new QuotaManager(1000, 800);
    const warnSpy = jest.spyOn(console, 'warn').mockImplementation(() => {});
    q.reserveToken(300, 'tensor', '');
    q.reserveToken(600, 'tensor', ''); // Exceeds 800
    expect(warnSpy).toHaveBeenCalled();
    warnSpy.mockRestore();
  });
});
