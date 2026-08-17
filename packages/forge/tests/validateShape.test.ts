import { validateShape } from "../src/tensor/validateShape";
import { AMEVAForgeShapeError } from "../src/errors";

describe("validateShape", () => {
  it("passes valid shape", () => {
    expect(validateShape([256, 256], "float32")).toBe(65536);
  });

  it("passes scalar empty shape", () => {
    expect(validateShape([], "float32")).toBe(1);
  });

  it("fails on 9D shape", () => {
    expect(() => validateShape([1, 2, 3, 4, 5, 6, 7, 8, 9], "float32")).toThrow(AMEVAForgeShapeError);
  });

  it("fails on negative dimension", () => {
    expect(() => validateShape([256, -256], "float32")).toThrow(AMEVAForgeShapeError);
  });

  it("fails on non-integer dimension", () => {
    expect(() => validateShape([256.5, 256], "float32")).toThrow(AMEVAForgeShapeError);
  });

  it("fails on byteLength mismatch", () => {
    // 10 elements * 4 bytes = 40 bytes. Pass 41 to fail.
    expect(() => validateShape([10], "float32", 41)).toThrow(AMEVAForgeShapeError);
  });

  it("fails on MAX_ELEMENTS exceeded", () => {
    // MAX_ELEMENTS = 256 * 1024 * 1024
    expect(() => validateShape([16384, 16385], "float32")).toThrow(AMEVAForgeShapeError);
  });
});
