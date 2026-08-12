import { AMEVAForgeDTypeError } from "../errors";
import { DType } from "../types";

export function validateDType(dtype: string): asserts dtype is DType {
  if (dtype !== "float32") {
    throw new AMEVAForgeDTypeError(`Unsupported dtype: ${dtype}. Only float32 is supported.`);
  }
}
