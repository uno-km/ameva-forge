import { 
  assertAllowedKernelName, 
  assertSafeShaderIdentifier, 
  assertStaticShaderSourceOnly, 
} from "../src/webgpu/shaderGuard";
import { AMEVAForgeSecurityError } from "../src/errors";

describe("shaderGuard", () => {
  it("allowed kernel name passes", () => {
    expect(() => assertAllowedKernelName("matmul")).not.toThrow();
  });

  it("unknown kernel name fails", () => {
    expect(() => assertAllowedKernelName("unknown_kernel")).toThrow(AMEVAForgeSecurityError);
  });

  it("safe identifier passes", () => {
    expect(() => assertSafeShaderIdentifier("valid_id_1")).not.toThrow();
  });

  it("unsafe identifier fails", () => {
    expect(() => assertSafeShaderIdentifier("invalid id!")).toThrow(AMEVAForgeSecurityError);
  });

  it("dynamic shader source interpolation is detected", () => {
    expect(() => assertStaticShaderSourceOnly("var a = ${val};")).toThrow(AMEVAForgeSecurityError);
    expect(() => assertStaticShaderSourceOnly("var a = `val`;")).toThrow(AMEVAForgeSecurityError);
  });
});
