import { 
  assertAllowedKernelName, 
  assertSafeShaderIdentifier, 
  assertStaticShaderSourceOnly, 
} from "../src/webgpu/shaderGuard";
import { AMEVATensorSecurityError } from "../src/errors";

describe("shaderGuard", () => {
  it("allowed kernel name passes", () => {
    expect(() => assertAllowedKernelName("matmul")).not.toThrow();
  });

  it("unknown kernel name fails", () => {
    expect(() => assertAllowedKernelName("unknown_kernel")).toThrow(AMEVATensorSecurityError);
  });

  it("safe identifier passes", () => {
    expect(() => assertSafeShaderIdentifier("valid_id_1")).not.toThrow();
  });

  it("unsafe identifier fails", () => {
    expect(() => assertSafeShaderIdentifier("invalid id!")).toThrow(AMEVATensorSecurityError);
  });

  it("dynamic shader source interpolation is detected", () => {
    expect(() => assertStaticShaderSourceOnly("var a = ${val};")).toThrow(AMEVATensorSecurityError);
    expect(() => assertStaticShaderSourceOnly("var a = `val`;")).toThrow(AMEVATensorSecurityError);
  });
});
