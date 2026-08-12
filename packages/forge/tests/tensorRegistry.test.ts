import { _globalRegistry } from "../src/tensor/tensorRegistry";
import { AMEVAForgeDisposedError } from "../src/errors";
import * as buffers from "../src/webgpu/buffers";

// Mock the webgpu buffers freeBuffer so we don't try to access navigator.gpu in tests
jest.mock("../src/webgpu/buffers", () => ({
  freeBuffer: jest.fn()
}));

describe("TensorRegistry", () => {
  beforeEach(() => {
    _globalRegistry.clear();
    jest.clearAllMocks();
  });

  it("register and get normally", () => {
    const handle = _globalRegistry.register({
      shape: [2, 2],
      dtype: "float32",
      byteLength: 16,
      buffer: {} as GPUBuffer
    });
    const record = _globalRegistry.get(handle);
    expect(record.handle).toBe(handle);
    expect(record.shape).toEqual([2, 2]);
  });

  it("fails on missing handle", () => {
    expect(() => _globalRegistry.get("invalid_handle")).toThrow(AMEVAForgeDisposedError);
  });

  it("fails on disposed handle access", () => {
    const handle = _globalRegistry.register({
      shape: [2, 2],
      dtype: "float32",
      byteLength: 16,
      buffer: {} as GPUBuffer
    });
    _globalRegistry.dispose(handle);
    expect(() => _globalRegistry.get(handle)).toThrow(AMEVAForgeDisposedError);
  });

  it("dispose idempotency", () => {
    const handle = _globalRegistry.register({
      shape: [2, 2],
      dtype: "float32",
      byteLength: 16,
      buffer: {} as GPUBuffer
    });
    _globalRegistry.dispose(handle);
    // Disposing twice should not throw
    expect(() => _globalRegistry.dispose(handle)).not.toThrow();
  });

  it("clear normally", () => {
    _globalRegistry.register({
      shape: [2, 2],
      dtype: "float32",
      byteLength: 16,
      buffer: {} as GPUBuffer
    });
    _globalRegistry.clear();
    expect(buffers.freeBuffer).toHaveBeenCalledTimes(1);
  });
});
