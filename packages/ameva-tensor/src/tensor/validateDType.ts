import { AMEVATensorDTypeError } from "../errors";
import { DType } from "../types";

export function validateDType(dtype: string): asserts dtype is DType {
  if (dtype !== "float32") {
    throw new AMEVATensorDTypeError(`Unsupported dtype: ${dtype}. Only float32 is supported.`);
  }
}
