import { assertWasmRange } from "../src/webgpu/validateWasmRange";
import { AMEVATensorSecurityError } from "../src/errors";

describe("validateWasmRange", () => {
  const wasmLength = 1000;

  it("passes normal range", () => {
    expect(() => assertWasmRange(0, 100, wasmLength)).not.toThrow();
    expect(() => assertWasmRange(500, 500, wasmLength)).not.toThrow();
  });

  it("fails on negative offset", () => {
    expect(() => assertWasmRange(-10, 100, wasmLength)).toThrow(AMEVATensorSecurityError);
  });

  it("fails on negative byteLength", () => {
    expect(() => assertWasmRange(10, -100, wasmLength)).toThrow(AMEVATensorSecurityError);
  });

  it("fails on unsafe integer", () => {
    expect(() => assertWasmRange(10.5, 100, wasmLength)).toThrow(AMEVATensorSecurityError);
  });

  it("fails on out-of-bounds", () => {
    expect(() => assertWasmRange(900, 200, wasmLength)).toThrow(AMEVATensorSecurityError);
  });

  it("boundary test for byteLength > wasmByteLength - offset", () => {
    expect(() => assertWasmRange(800, 200, wasmLength)).not.toThrow();
    expect(() => assertWasmRange(800, 201, wasmLength)).toThrow(AMEVATensorSecurityError);
  });
});
