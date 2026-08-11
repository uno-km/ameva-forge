import { AMEVATensorSecurityError } from "../errors";

export function assertWasmRange(offset: number, byteLength: number, wasmByteLength: number): void {
  if (!Number.isSafeInteger(offset) || offset < 0) {
    throw new AMEVATensorSecurityError("Invalid offset: must be a non-negative safe integer.");
  }
  if (!Number.isSafeInteger(byteLength) || byteLength < 0) {
    throw new AMEVATensorSecurityError("Invalid byteLength: must be a non-negative safe integer.");
  }
  if (!Number.isSafeInteger(wasmByteLength) || wasmByteLength < 0) {
    throw new AMEVATensorSecurityError("Invalid wasmByteLength: must be a non-negative safe integer.");
  }

  if (offset > wasmByteLength || byteLength > wasmByteLength - offset) {
    throw new AMEVATensorSecurityError("WASM memory range out of bounds");
  }
}
