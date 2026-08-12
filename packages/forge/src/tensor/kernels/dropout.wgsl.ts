export const DROPOUT_WGSL = `
struct Params {
  num_elements: u32,
  seed: f32,
  p: f32,
  padding: f32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32(hash: u32) -> f32 {
    return f32(hash) / 4294967295.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x + global_id.y * 65535u * 64u;
    if (index >= params.num_elements) {
        return;
    }
    
    let hash = pcg_hash(index + u32(params.seed * 10000.0));
    let rand = rand_f32(hash);
    
    if (rand < params.p) {
        out[index] = 0.0;
    } else {
        out[index] = x[index] * (1.0 / (1.0 - params.p));
    }
}
`;
