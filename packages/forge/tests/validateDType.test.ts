import { validateDType } from "../src/tensor/validateDType";
import { AMEVAForgeDTypeError } from "../src/errors";

describe("validateDType", () => {
  it("passes float32", () => {
    expect(() => validateDType("float32")).not.toThrow();
  });

  it("fails on float16", () => {
    expect(() => validateDType("float16")).toThrow(AMEVAForgeDTypeError);
  });

  it("fails on int32", () => {
    expect(() => validateDType("int32")).toThrow(AMEVAForgeDTypeError);
  });

  it("fails on unknown dtype", () => {
    expect(() => validateDType("unknown_type" as any)).toThrow(AMEVAForgeDTypeError);
  });
});
