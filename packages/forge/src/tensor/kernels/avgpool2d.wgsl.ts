export const AVGPOOL2D_WGSL = `
struct Params {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    kH: u32,
    kW: u32,
    sH: u32,
    sW: u32,
    pH: u32,
    pW: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch * params.channels * params.out_h * params.out_w;
    if (idx >= total) {
        return;
    }
    
    let ow = idx % params.out_w;
    let oh = (idx / params.out_w) % params.out_h;
    let c = (idx / (params.out_w * params.out_h)) % params.channels;
    let b = idx / (params.out_w * params.out_h * params.channels);
    
    let h_start = i32(oh * params.sH) - i32(params.pH);
    let w_start = i32(ow * params.sW) - i32(params.pW);
    
    var sum = 0.0;
    var count = 0.0;
    
    for (var kh = 0u; kh < params.kH; kh++) {
        for (var kw = 0u; kw < params.kW; kw++) {
            let h = h_start + i32(kh);
            let w = w_start + i32(kw);
            
            if (h >= 0 && h < i32(params.in_h) && w >= 0 && w < i32(params.in_w)) {
                let in_idx = ((b * params.channels + c) * params.in_h + u32(h)) * params.in_w + u32(w);
                sum += input[in_idx];
                count += 1.0;
            }
        }
    }
    
    if (count > 0.0) {
        output[idx] = sum / count;
    } else {
        output[idx] = 0.0;
    }
}
`;
