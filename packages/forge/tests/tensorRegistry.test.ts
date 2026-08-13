import { _globalRegistry } from "../src/tensor/tensorRegistry";
import { AMEVAForgeDisposedError } from "../src/errors";
import * as buffers from "../src/webgpu/buffers";
import { AllocationToken } from "../src/webgpu/quota";

// Mock the webgpu buffers freeBuffer so we don't try to access navigator.gpu in tests
jest.mock("../src/webgpu/buffers", () => ({
  freeBuffer: jest.fn()
}));

describe("TensorRegistry", () => {
  beforeEach(() => {
    _globalRegistry.clear();
    jest.clearAllMocks();
  });

  const dummyToken = new AllocationToken("test", 16, "tensor", null, 0);

  it("register and get normally", () => {
    const handle = _globalRegistry.register({
      shape: [2, 2],
      dtype: "float32",
      byteLength: 16,
      buffer: {} as GPUBuffer,
      token: dummyToken
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
      buffer: {} as GPUBuffer,
      token: dummyToken
    });
    _globalRegistry.dispose(handle);
    expect(() => _globalRegistry.get(handle)).toThrow(AMEVAForgeDisposedError);
  });

  it("dispose idempotency", () => {
    const handle = _globalRegistry.register({
      shape: [2, 2],
      dtype: "float32",
      byteLength: 16,
      buffer: {} as GPUBuffer,
      token: dummyToken
    });
    _globalRegistry.dispose(handle);
    // Disposing twice should not throw
    expect(() => _globalRegistry.dispose(handle)).not.toThrow();
  });

  it("clear normally", () => {
    const destroyMock = jest.fn();
    _globalRegistry.register({
      shape: [2, 2],
      dtype: "float32",
      byteLength: 16,
      buffer: { destroy: destroyMock } as unknown as GPUBuffer,
      token: dummyToken
    });
    _globalRegistry.clear();
    // When device is not present, it calls _safeDestroyBuffer which calls buffer.destroy()
    expect(destroyMock).toHaveBeenCalledTimes(1);
  });
});
