/**
 * 파일 생성일: 2026-08-18
 * AMEVA-Forge Release 2.0: SCRUM-221 SwiGLU Fused Activation WGSL Kernel
 *
 * WHAT: Swish Gated Linear Unit (SwiGLU) 융합 활성화 함수 WGSL 셰이더입니다.
 * WHY: LLaMA 및 Gemma 등의 FFN 블록에서 Gate Projection(x)과 Up Projection(y)의 원소별 Swish 게이팅을
 *      중간 메모리 왕복 없이 단일 커널로 초고속 처리하기 위해 존재합니다.
 * HOW: Swish(x) = x * sigmoid(x) = x / (1.0 + exp(-x)) 연산 후 y와 원소별 곱셈을 수행합니다.
 */
export declare const SWIGLU_WGSL = "\nstruct Params {\n  num_elements: u32,  // \uCD1D \uC6D0\uC18C \uAC1C\uC218\n  workgroupsX: u32,   // 2D \uB514\uC2A4\uD328\uCE58 X \uD06C\uAE30\n  pad1: u32,\n  pad2: u32,\n};\n\n@group(0) @binding(0) var<uniform> params: Params;\n@group(0) @binding(1) var<storage, read> gate: array<f32>; // Gate projection (x)\n@group(0) @binding(2) var<storage, read> up: array<f32>;   // Up projection (y)\n@group(0) @binding(3) var<storage, read_write> out: array<f32>; // SwiGLU output\n\n@compute @workgroup_size(64, 1, 1)\nfn main(\n  @builtin(local_invocation_id) local_id: vec3<u32>,\n  @builtin(workgroup_id) workgroup_id: vec3<u32>\n) {\n  let idx = (workgroup_id.x + workgroup_id.y * params.workgroupsX) * 64u + local_id.x;\n\n  if (idx >= params.num_elements) {\n    return;\n  }\n\n  let x = gate[idx];\n  let y = up[idx];\n\n  // Swish(x) = x / (1.0 + exp(-x))\n  let swish_x = x / (1.0 + exp(-x));\n  out[idx] = swish_x * y;\n}\n";
