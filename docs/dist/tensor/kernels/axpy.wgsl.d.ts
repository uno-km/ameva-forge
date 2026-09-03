/**
 * 생성일 (Created): 2026-08-12 12:14:52 +0900
 * 수정 내역 (Modified):
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *   - 2026-08-18 14:10:00 +0900: Pure Standard IEEE-754 SGD Update without silent NaN/Inf zeroing
 */
export declare const AXPY_WGSL = "\n/**\n * @struct Params\n * @brief AXPY (param = param - lr * grad) \uC5F0\uC0B0 \uD30C\uB77C\uBBF8\uD130 \uAD6C\uC870\uCCB4\n */\nstruct Params {\n  numElements: u32,\n  lr: f32,\n  workgroups_x: u32,\n  pad1: u32,\n};\n\n@group(0) @binding(0) var<uniform> params: Params;\n@group(0) @binding(1) var<storage, read> grad: array<f32>;\n@group(0) @binding(2) var<storage, read_write> param: array<f32>;\n\n@compute @workgroup_size(64)\nfn main(@builtin(global_invocation_id) global_id: vec3<u32>) {\n  let idx = global_id.x + global_id.y * params.workgroups_x * 64u;\n  if (idx >= params.numElements) {\n    return;\n  }\n  \n  let g = grad[idx];\n  // Standard SGD in-place update (IEEE 754 float32)\n  param[idx] = param[idx] - params.lr * g;\n}\n";
