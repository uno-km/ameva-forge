import { QuotaManager } from "../src/webgpu/quota";
import { AMEVATensorQuotaExceededError } from "../src/errors";

describe("QuotaManager", () => {
  it("normal reserve and release", () => {
    const q = new QuotaManager(1000, 500);
    q.reserve(300);
    expect(q.getUsage().allocatedBytes).toBe(300);
    q.release(100);
    expect(q.getUsage().allocatedBytes).toBe(200);
  });

  it("fails on hard limit exceeded", () => {
    const q = new QuotaManager(1000, 500);
    expect(() => q.reserve(1001)).toThrow(AMEVATensorQuotaExceededError);
  });

  it("fails on invalid byteLength", () => {
    const q = new QuotaManager(1000, 500);
    expect(() => q.reserve(-100)).toThrow(AMEVATensorQuotaExceededError);
    expect(() => q.reserve(10.5)).toThrow(AMEVATensorQuotaExceededError);
  });

  it("fails on release over allocated", () => {
    const q = new QuotaManager(1000, 500);
    q.reserve(300);
    expect(() => q.release(400)).toThrow(AMEVATensorQuotaExceededError);
  });

  it("reset works properly", () => {
    const q = new QuotaManager(1000, 500);
    q.reserve(300);
    q.reset();
    expect(q.getUsage().allocatedBytes).toBe(0);
  });
});
