import { test, expect } from '@playwright/test';

const ABS_TOL = 1e-4;

function allClose(actual: number[], expected: number[]): boolean {
  if (actual.length !== expected.length) return false;
  for (let i = 0; i < actual.length; i++) {
    if (Math.abs(actual[i] - expected[i]) > ABS_TOL) return false;
  }
  return true;
}

test('V3 Extended Kernels (Rectangular BMM, Scatter, Gather, Permute) in WebGPU and Pyodide', async ({ page }) => {
  await page.goto('/test.html');
  const gpuAvailable = await page.evaluate(async () => {
    if (typeof navigator === 'undefined' || !('gpu' in navigator)) return false;
    try {
      const adapter = await navigator.gpu.requestAdapter();
      return adapter !== null;
    } catch {
      return false;
    }
  });

  const releaseGate = test.info().project.name === 'release-webgpu';
  if (!gpuAvailable && releaseGate) {
    throw new Error('RELEASE_GATE_WEBGPU_UNAVAILABLE: A physical WebGPU adapter is required.');
  }
  if (!gpuAvailable) {
    test.skip(!gpuAvailable, 'WebGPU adapter is unavailable');
  }

  // ── PART 1: JS WebGPU Kernels with Rectangular Dimensions (Non-Square M != N != K) ──
  const jsResult = await page.evaluate(async () => {
    const forge = (window as any).forge;
    if (!forge) throw new Error('Forge browser API unavailable');
    await forge.initWebGPU();

    const handles: string[] = [];

    // 1. Rectangular Batched Matmul Test:
    // B=2, M=2, K=3, N=2 -> A is [2, 2, 3], B is [2, 3, 2] -> C is [2, 2, 2]
    // Batch 0:
    // A0 = [[1, 2, 3], [4, 5, 6]] (2x3)
    // B0 = [[1, 0], [0, 1], [1, 1]] (3x2)
    // C0 = [[1*1+2*0+3*1, 1*0+2*1+3*1], [4*1+5*0+6*1, 4*0+5*1+6*1]] = [[4, 5], [10, 11]]
    // Batch 1:
    // A1 = [[2, 1, 0], [0, 1, 2]] (2x3)
    // B1 = [[2, 1], [1, 0], [0, 2]] (3x2)
    // C1 = [[2*2+1*1+0*0, 2*1+1*0+0*2], [0*2+1*1+2*0, 0*1+1*0+2*2]] = [[5, 2], [1, 4]]
    const bmm_a_data = new Float32Array([1, 2, 3, 4, 5, 6, 2, 1, 0, 0, 1, 2]); // [2, 2, 3]
    const bmm_b_data = new Float32Array([1, 0, 0, 1, 1, 1, 2, 1, 1, 0, 0, 2]); // [2, 3, 2]
    const h_bmm_a = forge.uploadFloat32Array(bmm_a_data, [2, 2, 3]);
    const h_bmm_b = forge.uploadFloat32Array(bmm_b_data, [2, 3, 2]);
    handles.push(h_bmm_a, h_bmm_b);

    // params: [B=2, M=2, N=2, K=3]
    const bmm_graph = [
      { op: 'load', id: 1, handle: h_bmm_a, shape: [2, 2, 3] },
      { op: 'load', id: 2, handle: h_bmm_b, shape: [2, 3, 2] },
      { op: 'batched_matmul', id: 3, shape: [2, 2, 2], in: [1, 2], params: [2, 2, 2, 3] },
    ];

    const bmm_out = await forge.executeGraph(JSON.stringify(bmm_graph), []);
    const h_bmm_res = bmm_out[3];
    handles.push(h_bmm_res);
    await forge.mapBufferAsync(h_bmm_res);
    const bmm_actual = new Float32Array(8);
    forge.readMappedInto(h_bmm_res, bmm_actual);
    const expected_bmm = [4, 5, 10, 11, 5, 2, 1, 4];

    // 2. Rectangular Scatter Test with Base Tensor & Negative Indices:
    // x = [[1, 2, 3, 4], [5, 6, 7, 8]] (2x4), dim=1, index=[[0, -1], [1, -2]], src=[[99, 88], [77, 66]]
    // dim_size=4. In row 0: -1 -> 3. row 0 col 0 becomes 99, row 0 col 3 becomes 88.
    // In row 1: -2 -> 2. row 1 col 1 becomes 77, row 1 col 2 becomes 66.
    // Result: [[99, 2, 3, 88], [5, 77, 66, 8]]
    const scat_x_data = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const scat_idx_data = new Float32Array([0, -1, 1, -2]);
    const scat_src_data = new Float32Array([99, 88, 77, 66]);
    const h_scat_x = forge.uploadFloat32Array(scat_x_data, [2, 4]);
    const h_scat_idx = forge.uploadFloat32Array(scat_idx_data, [2, 2]);
    const h_scat_src = forge.uploadFloat32Array(scat_src_data, [2, 2]);
    handles.push(h_scat_x, h_scat_idx, h_scat_src);

    const scat_params = [
      4, 1, 2, 0,
      4, 1, 0, 0, 0, 0, 0, 0,
      2, 1, 0, 0, 0, 0, 0, 0,
      2, 4, 0, 0, 0, 0, 0, 0
    ];

    const scat_graph = [
      { op: 'load', id: 1, handle: h_scat_idx, shape: [2, 2] },
      { op: 'load', id: 2, handle: h_scat_src, shape: [2, 2] },
      { op: 'load', id: 3, handle: h_scat_x, shape: [2, 4] },
      { op: 'scatter', id: 4, shape: [2, 4], in: [1, 2, 3], params: scat_params },
    ];

    const scat_out = await forge.executeGraph(JSON.stringify(scat_graph), []);
    const h_scat_res = scat_out[4];
    handles.push(h_scat_res);
    await forge.mapBufferAsync(h_scat_res);
    const scat_actual = new Float32Array(8);
    forge.readMappedInto(h_scat_res, scat_actual);
    const expected_scat = [99, 2, 3, 88, 5, 77, 66, 8];

    // 3. Rectangular Gather Test with Negative Index:
    // x = [[10, 20, 30, 40], [50, 60, 70, 80]], dim=1, index=[[-1, 0], [2, -2]]
    // row 0: -1 -> 40, 0 -> 10. row 1: 2 -> 70, -2 -> 70
    // Result: [[40, 10], [70, 70]]
    const gat_x_data = new Float32Array([10, 20, 30, 40, 50, 60, 70, 80]);
    const gat_idx_data = new Float32Array([-1, 0, 2, -2]);
    const h_gat_x = forge.uploadFloat32Array(gat_x_data, [2, 4]);
    const h_gat_idx = forge.uploadFloat32Array(gat_idx_data, [2, 2]);
    handles.push(h_gat_x, h_gat_idx);

    const gat_params = [
      4, 1, 2, 0,
      4, 1, 0, 0, 0, 0, 0, 0,
      2, 1, 0, 0, 0, 0, 0, 0,
      2, 4, 0, 0, 0, 0, 0, 0
    ];

    const gat_graph = [
      { op: 'load', id: 1, handle: h_gat_x, shape: [2, 4] },
      { op: 'load', id: 2, handle: h_gat_idx, shape: [2, 2] },
      { op: 'gather', id: 3, shape: [2, 2], in: [1, 2], params: gat_params },
    ];

    const gat_out = await forge.executeGraph(JSON.stringify(gat_graph), []);
    const h_gat_res = gat_out[3];
    handles.push(h_gat_res);
    await forge.mapBufferAsync(h_gat_res);
    const gat_actual = new Float32Array(4);
    forge.readMappedInto(h_gat_res, gat_actual);
    const expected_gat = [40, 10, 70, 70];

    const h_da = forge.uploadFloat32Array(new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]);
    const h_db = forge.uploadFloat32Array(new Float32Array([10, 20, 30, 40, 50, 60]), [2, 3]);
    const addFn = (forge.gpuCore && forge.gpuCore.add) || forge.add;
    const mulFn = (forge.gpuCore && forge.gpuCore.mul) || forge.mul;
    const h_dadd = await addFn(h_da, h_db);
    const h_dmul = await mulFn(h_da, h_db);
    handles.push(h_da, h_db, h_dadd, h_dmul);
    await forge.mapBufferAsync(h_dadd);
    const dadd_actual = new Float32Array(6);
    forge.readMappedInto(h_dadd, dadd_actual);
    await forge.mapBufferAsync(h_dmul);
    const dmul_actual = new Float32Array(6);
    forge.readMappedInto(h_dmul, dmul_actual);
    const expected_dadd = [11, 22, 33, 44, 55, 66];
    const expected_dmul = [10, 40, 90, 160, 250, 360];

    // 5. Large 4.3M Tensor Direct API 2D Dispatch Execution (N=4,300,000 > 4.19M):
    const largeN = 4_300_000;
    const largeA = new Float32Array(largeN);
    const largeB = new Float32Array(largeN);
    largeA.fill(1.0);
    largeB.fill(2.0);
    const h_la = forge.uploadFloat32Array(largeA, [largeN]);
    const h_lb = forge.uploadFloat32Array(largeB, [largeN]);
    const h_ladd = await addFn(h_la, h_lb);
    const reluFn = (forge.gpuCore && forge.gpuCore.relu) || forge.relu;
    const reluBwdFn = (forge.gpuCore && forge.gpuCore.relu_backward) || forge.relu_backward;
    const h_lfwd_relu = await reluFn(h_la);
    const h_lrelu = await reluBwdFn(h_la, h_lb);
    handles.push(h_la, h_lb, h_ladd, h_lfwd_relu, h_lrelu);
    await forge.mapBufferAsync(h_ladd);
    const ladd_actual = new Float32Array(largeN);
    forge.readMappedInto(h_ladd, ladd_actual);
    await forge.mapBufferAsync(h_lfwd_relu);
    const lfwd_relu_actual = new Float32Array(largeN);
    forge.readMappedInto(h_lfwd_relu, lfwd_relu_actual);
    await forge.mapBufferAsync(h_lrelu);
    const lrelu_actual = new Float32Array(largeN);
    forge.readMappedInto(h_lrelu, lrelu_actual);
    const large_samples = [
      ladd_actual[0],
      ladd_actual[4_194_239],
      ladd_actual[4_194_240],
      ladd_actual[largeN - 1],
    ];
    const relu_fwd_samples = [
      lfwd_relu_actual[0],
      lfwd_relu_actual[4_194_239],
      lfwd_relu_actual[4_194_240],
      lfwd_relu_actual[largeN - 1],
    ];
    const relu_bwd_samples = [
      lrelu_actual[0],
      lrelu_actual[4_194_239],
      lrelu_actual[4_194_240],
      lrelu_actual[largeN - 1],
    ];

    // 6. Large 4.3M Tensor Permute 2D Dispatch Execution (Shape [4300, 1000] -> permute [1, 0] -> [1000, 4300]):
    const pRows = 4300;
    const pCols = 1000;
    const pTotal = pRows * pCols; // 4,300,000 > 4.19M
    const pData = new Float32Array(pTotal);
    for (let r = 0; r < pRows; r++) {
      for (let c = 0; c < pCols; c++) {
        pData[r * pCols + c] = r * 10000 + c;
      }
    }
    const h_pin = forge.uploadFloat32Array(pData, [pRows, pCols]);
    const pInstructions = [
      { op: 'load', id: 1, handle: h_pin, shape: [pRows, pCols], in: [], params: [] },
      { op: 'permute', id: 2, shape: [pCols, pRows], in: [1], params: [1, 0] }
    ];
    const pOutMap = await forge.executeGraph(JSON.stringify(pInstructions), [], [2]);
    const h_pout = pOutMap[2];
    handles.push(h_pin, h_pout);
    await forge.mapBufferAsync(h_pout);
    const pout_actual = new Float32Array(pTotal);
    forge.readMappedInto(h_pout, pout_actual);
    
    const permute_samples = [
      pout_actual[0 * pRows + 0],
      pout_actual[0 * pRows + 4299],
      pout_actual[999 * pRows + 0],
      pout_actual[999 * pRows + 4299],
    ];
    const expected_permute_samples = [
      pData[0 * pCols + 0],
      pData[4299 * pCols + 0],
      pData[0 * pCols + 999],
      pData[4299 * pCols + 999],
    ];

    // 7. Dropout Deterministic Seed Reproducibility Test:
    const dropData = new Float32Array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    const h_drop_in = forge.uploadFloat32Array(dropData, [8]);
    const dropInst1 = [
      { op: 'load', id: 1, handle: h_drop_in, shape: [8], in: [], params: [] },
      { op: 'dropout', id: 2, shape: [8], in: [1], params: [42, 0.5] }
    ];
    const dropInst2 = [
      { op: 'load', id: 1, handle: h_drop_in, shape: [8], in: [], params: [] },
      { op: 'dropout', id: 2, shape: [8], in: [1], params: [42, 0.5] }
    ];
    const outDrop1 = await forge.executeGraph(JSON.stringify(dropInst1), [], [2]);
    const outDrop2 = await forge.executeGraph(JSON.stringify(dropInst2), [], [2]);
    const h_d1 = outDrop1[2];
    const h_d2 = outDrop2[2];
    handles.push(h_drop_in, h_d1, h_d2);
    await forge.mapBufferAsync(h_d1);
    const d1_actual = new Float32Array(8);
    forge.readMappedInto(h_d1, d1_actual);
    await forge.mapBufferAsync(h_d2);
    const d2_actual = new Float32Array(8);
    forge.readMappedInto(h_d2, d2_actual);

    // 8. Scatter NaN index skip test:
    const scatBase = new Float32Array([10.0, 20.0, 30.0]);
    const scatSrc = new Float32Array([99.0]);
    const scatNanIdx = new Float32Array([NaN]);
    const h_sb = forge.uploadFloat32Array(scatBase, [3]);
    const h_ss = forge.uploadFloat32Array(scatSrc, [1]);
    const h_sn = forge.uploadFloat32Array(scatNanIdx, [1]);
    const scatNanInst = [
      { op: 'load', id: 1, handle: h_sb, shape: [3], in: [], params: [] },
      { op: 'load', id: 2, handle: h_ss, shape: [1], in: [], params: [] },
      { op: 'load', id: 3, handle: h_sn, shape: [1], in: [], params: [] },
      { op: 'scatter', id: 4, shape: [3], in: [3, 2, 1], params: [1, 0, 1, 1, 0, 1, 0, 1, 0, 3, 0] }
    ];
    const outScatNan = await forge.executeGraph(JSON.stringify(scatNanInst), [], [4]);
    const h_sout = outScatNan[4];
    handles.push(h_sb, h_ss, h_sn, h_sout);
    await forge.mapBufferAsync(h_sout);
    const scat_nan_actual = new Float32Array(3);
    forge.readMappedInto(h_sout, scat_nan_actual);

    // 9. Matmul Tiled 16x16 Test:
    const tM = 32, tN = 32, tK = 32;
    const matA = new Float32Array(tM * tK);
    const matB = new Float32Array(tK * tN);
    for (let i = 0; i < matA.length; i++) matA[i] = (i % 7) * 0.1;
    for (let i = 0; i < matB.length; i++) matB[i] = (i % 5) * 0.1;
    const h_matA = forge.uploadFloat32Array(matA, [tM, tK]);
    const h_matB = forge.uploadFloat32Array(matB, [tK, tN]);
    const matTiledInst = [
      { op: 'load', id: 1, handle: h_matA, shape: [tM, tK], in: [], params: [] },
      { op: 'load', id: 2, handle: h_matB, shape: [tK, tN], in: [], params: [] },
      { op: 'matmul_tiled', id: 3, shape: [tM, tN], in: [1, 2], params: [tM, tN, tK] }
    ];
    const outMatTiled = await forge.executeGraph(JSON.stringify(matTiledInst), [], [3]);
    const h_mt = outMatTiled[3];
    handles.push(h_matA, h_matB, h_mt);
    await forge.mapBufferAsync(h_mt);
    const mt_actual = new Float32Array(tM * tN);
    forge.readMappedInto(h_mt, mt_actual);

    const mt_expected = new Float32Array(tM * tN);
    for (let r = 0; r < tM; r++) {
      for (let c = 0; c < tN; c++) {
        let sum = 0;
        for (let k = 0; k < tK; k++) {
          sum += matA[r * tK + k] * matB[k * tN + c];
        }
        mt_expected[r * tN + c] = sum;
      }
    }

    // 10. Native Embedding Forward & Backward Test with non-zero indices [2, 0, 1]:
    const embVocab = 4, embDim = 8;
    const embWeightData = new Float32Array(embVocab * embDim);
    for (let v = 0; v < embVocab; v++) {
      for (let d = 0; d < embDim; d++) {
        embWeightData[v * embDim + d] = (v + 1) * 10 + d;
      }
    }
    const embIdxData = new Float32Array([2, 0, 1]); // non-zero tokens!
    const h_ew = forge.uploadFloat32Array(embWeightData, [embVocab, embDim]);
    const h_ei = forge.uploadFloat32Array(embIdxData, [3]);
    const embInst = [
      { op: 'load', id: 1, handle: h_ew, shape: [embVocab, embDim], in: [], params: [] },
      { op: 'load', id: 2, handle: h_ei, shape: [3], in: [], params: [] },
      { op: 'embedding', id: 3, shape: [3, embDim], in: [1, 2], params: [3, embDim, embVocab, 0] }
    ];
    const outEmb = await forge.executeGraph(JSON.stringify(embInst), [], [3]);
    const h_eout = outEmb[3];
    handles.push(h_ew, h_ei, h_eout);
    await forge.mapBufferAsync(h_eout);
    const emb_actual = new Float32Array(3 * embDim);
    forge.readMappedInto(h_eout, emb_actual);

    for (const h of handles) {
      try { forge.dispose(h); } catch {}
    }
    forge.flushGC();

    return {
      bmm_actual: Array.from(bmm_actual),
      expected_bmm,
      scat_actual: Array.from(scat_actual),
      expected_scat,
      gat_actual: Array.from(gat_actual),
      expected_gat,
      dadd_actual: Array.from(dadd_actual),
      expected_dadd,
      dmul_actual: Array.from(dmul_actual),
      expected_dmul,
      large_samples,
      relu_fwd_samples,
      relu_bwd_samples,
      permute_samples,
      expected_permute_samples,
      drop_reproducible: Array.from(d1_actual).every((v, i) => v === d2_actual[i]),
      scat_nan_unmodified: Array.from(scat_nan_actual),
      mt_actual: Array.from(mt_actual),
      mt_expected: Array.from(mt_expected),
      emb_actual: Array.from(emb_actual),
      emb_expected_row0: Array.from(embWeightData.slice(2 * embDim, 3 * embDim)),
      emb_expected_row1: Array.from(embWeightData.slice(0, embDim)),
      emb_expected_row2: Array.from(embWeightData.slice(embDim, 2 * embDim)),
    };
  });

  expect(allClose(jsResult.bmm_actual, jsResult.expected_bmm)).toBe(true);
  expect(allClose(jsResult.scat_actual, jsResult.expected_scat)).toBe(true);
  expect(allClose(jsResult.gat_actual, jsResult.expected_gat)).toBe(true);
  expect(allClose(jsResult.dadd_actual, jsResult.expected_dadd)).toBe(true);
  expect(allClose(jsResult.dmul_actual, jsResult.expected_dmul)).toBe(true);
  expect(jsResult.large_samples).toEqual([3, 3, 3, 3]);
  expect(jsResult.relu_fwd_samples).toEqual([1, 1, 1, 1]);
  expect(jsResult.relu_bwd_samples).toEqual([2, 2, 2, 2]);
  expect(allClose(jsResult.permute_samples, jsResult.expected_permute_samples)).toBe(true);
  expect(jsResult.drop_reproducible).toBe(true);
  expect(jsResult.scat_nan_unmodified).toEqual([10, 20, 30]);
  expect(allClose(jsResult.mt_actual, jsResult.mt_expected)).toBe(true);
  expect(allClose(jsResult.emb_actual.slice(0, 8), jsResult.emb_expected_row0)).toBe(true);
  expect(allClose(jsResult.emb_actual.slice(8, 16), jsResult.emb_expected_row1)).toBe(true);
  expect(allClose(jsResult.emb_actual.slice(16, 24), jsResult.emb_expected_row2)).toBe(true);

  // ── PART 2: Pyodide Python WebGPU E2E Verification ──
  const pyResult = await page.evaluate(async () => {
    await (window as any).__AMEVA_PYODIDE_READY__;
    const pyodide = (window as any).pyodide;
    if (!pyodide) {
      throw new Error('Pyodide failed to load: window.pyodide is unavailable');
    }

    const pyCode = `
import forge
import numpy as np

# 1. BMM in Python on WebGPU:
a_np = np.array([[[1, 2, 3], [4, 5, 6]], [[2, 1, 0], [0, 1, 2]]], dtype=np.float32)
b_np = np.array([[[1, 0], [0, 1], [1, 1]], [[2, 1], [1, 0], [0, 2]]], dtype=np.float32)

t_a = forge.tensor(a_np).to('gpu')
t_b = forge.tensor(b_np).to('gpu')
t_c = forge.ops.bmm(t_a, t_b)

res_bmm = await t_c.numpy_async()
expected_bmm = np.matmul(a_np, b_np)
bmm_diff = float(np.max(np.abs(res_bmm - expected_bmm)))

# 2. Scatter in Python on WebGPU:
x_np = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
idx_np = np.array([[0, -1], [1, -2]], dtype=np.float32)
src_np = np.array([[99, 88], [77, 66]], dtype=np.float32)

t_x = forge.tensor(x_np).to('gpu')
t_idx = forge.tensor(idx_np).to('gpu')
t_src = forge.tensor(src_np).to('gpu')
t_scat = forge.ops.scatter(t_x, 1, t_idx, t_src)

res_scat = await t_scat.numpy_async()
expected_scat = np.array([[99, 2, 3, 88], [5, 77, 66, 8]], dtype=np.float32)
scat_diff = float(np.max(np.abs(res_scat - expected_scat)))

# 3. Gather in Python on WebGPU:
t_gat = forge.ops.gather(t_x, 1, t_idx)
res_gat = await t_gat.numpy_async()
expected_gat = np.array([[1, 4], [6, 7]], dtype=np.float32)
gat_diff = float(np.max(np.abs(res_gat - expected_gat)))

# 4. Causal Attention on WebGPU:
q_np = np.ones((1, 4, 8), dtype=np.float32)
k_np = np.ones((1, 4, 8), dtype=np.float32)
v_np = np.eye(4, 8, dtype=np.float32).reshape(1, 4, 8)

t_q = forge.tensor(q_np).to('gpu')
t_k = forge.tensor(k_np).to('gpu')
t_v = forge.tensor(v_np).to('gpu')

t_out = forge.functional.scaled_dot_product_attention(t_q, t_k, t_v, is_causal=True)
res_causal = await t_out.numpy_async()
causal_diff = float(abs(res_causal[0, 0, 0] - 1.0) + np.max(np.abs(res_causal[0, 0, 1:])))

# 5. Softmax with Large Logits (>100.0) on WebGPU:
logits_np = np.array([[100.0, 105.0], [50.0, 52.0]], dtype=np.float32)
t_logits = forge.tensor(logits_np).to('gpu')
t_sm = forge.functional.softmax(t_logits, axis=-1)
res_sm = await t_sm.numpy_async()
sm_nan = bool(np.isnan(res_sm).any() or np.isinf(res_sm).any())
sm_sum_diff = float(np.max(np.abs(np.sum(res_sm, axis=-1) - 1.0)))

# 6. MaxAxis on WebGPU:
t_max = forge.ops.max_axis(t_x, axis=1)
res_max = await t_max.numpy_async()
expected_max = np.max(x_np, axis=1)
max_axis_diff = float(np.max(np.abs(res_max - expected_max)))

# 7. MaxAxis Backward on WebGPU Autograd:
t_x_grad = forge.tensor(np.array([[1.0, 5.0, 2.0], [4.0, 4.0, 1.0]], dtype=np.float32), requires_grad=True).to('gpu')
t_max_grad = forge.ops.max_axis(t_x_grad, axis=1)
loss = forge.ops.sum_op(t_max_grad)
loss.backward()
grad_x_res = await t_x_grad.grad.numpy_async()
expected_grad_x = np.array([[0.0, 1.0, 0.0], [0.5, 0.5, 0.0]], dtype=np.float32)
max_axis_backward_diff = float(np.max(np.abs(grad_x_res - expected_grad_x)))

# 8. 5D and 6D Broadcasting on WebGPU:
a_5d_np = np.ones((2, 1, 3, 1, 4), dtype=np.float32)
b_5d_np = np.full((1, 5, 1, 7, 4), 2.0, dtype=np.float32)
t_a_5d = forge.tensor(a_5d_np).to('gpu')
t_b_5d = forge.tensor(b_5d_np).to('gpu')
t_out_5d = t_a_5d + t_b_5d
res_5d = await t_out_5d.numpy_async()
expected_5d = a_5d_np + b_5d_np
broadcast_5d_diff = float(np.max(np.abs(res_5d - expected_5d)))

a_6d_np = np.ones((1, 2, 1, 3, 1, 4), dtype=np.float32)
b_6d_np = np.full((5, 1, 2, 1, 3, 4), 3.0, dtype=np.float32)
t_a_6d = forge.tensor(a_6d_np).to('gpu')
t_b_6d = forge.tensor(b_6d_np).to('gpu')
t_out_6d = t_a_6d * t_b_6d
res_6d = await t_out_6d.numpy_async()
expected_6d = a_6d_np * b_6d_np
broadcast_6d_diff = float(np.max(np.abs(res_6d - expected_6d)))

# 9. GPU SGD Momentum Native WebGPU Execution:
param_gpu = forge.tensor([1.0, 2.0], requires_grad=True).to('gpu')
param_gpu.grad = forge.tensor([0.1, 0.2]).to('gpu')
opt_mom = forge.optim.SGD([param_gpu], lr=0.1, momentum=0.9)
await opt_mom.step_async()
param_gpu_val = await param_gpu.numpy_async()
momentum_guard_passed = bool(abs(param_gpu_val[0] - 0.99) < 1e-3 and abs(param_gpu_val[1] - 1.98) < 1e-3)

# 10. Strict Mode NaN Gradient Detection on WebGPU:
param_strict = forge.tensor([1.0, 2.0], requires_grad=True).to('gpu')
t_zero = forge.tensor([0.0, 0.5]).to('gpu')
t_div = forge.tensor([0.0, 1.0]).to('gpu')
param_strict.grad = t_zero / t_div # GPU computation resulting in NaN at index 0
opt_strict = forge.optim.SGD([param_strict], lr=0.1, strict_training=True)
strict_nan_caught = False
try:
    await opt_strict.step_async()
except forge.errors.AMEVAForgeValidationError:
    strict_nan_caught = True

# 11. Non-Strict Mode IEEE-754 NaN Propagation on WebGPU:
param_non_strict = forge.tensor([1.0, 2.0], requires_grad=True).to('gpu')
param_non_strict.grad = t_zero / t_div
opt_non_strict = forge.optim.SGD([param_non_strict], lr=0.1, strict_training=False)
await opt_non_strict.step_async()
param_non_strict_val = await param_non_strict.numpy_async()
non_strict_nan_propagated = bool(np.isnan(param_non_strict_val[0]) and abs(param_non_strict_val[1] - 1.95) < 1e-4)

# 12. Full 8D Broadcasting Parity on WebGPU (Add, Mul, Div):
a_8d_np = np.ones((1, 1, 2, 1, 3, 1, 4, 1), dtype=np.float32)
b_8d_np = np.full((6, 1, 1, 5, 1, 7, 1, 8), 2.0, dtype=np.float32)
t_a_8d = forge.tensor(a_8d_np).to('gpu')
t_b_8d = forge.tensor(b_8d_np).to('gpu')
t_out_8d_add = t_a_8d + t_b_8d
t_out_8d_mul = t_a_8d * t_b_8d
t_out_8d_div = t_b_8d / t_a_8d
res_8d_add = await t_out_8d_add.numpy_async()
res_8d_mul = await t_out_8d_mul.numpy_async()
res_8d_div = await t_out_8d_div.numpy_async()
broadcast_8d_add_diff = float(np.max(np.abs(res_8d_add - (a_8d_np + b_8d_np))))
broadcast_8d_mul_diff = float(np.max(np.abs(res_8d_mul - (a_8d_np * b_8d_np))))
broadcast_8d_div_diff = float(np.max(np.abs(res_8d_div - (b_8d_np / a_8d_np))))
broadcast_8d_diff = max(broadcast_8d_add_diff, broadcast_8d_mul_diff, broadcast_8d_div_diff)

# 13. BatchNorm2d Train -> Eval Transition on WebGPU:
bn_gpu = forge.nn.BatchNorm2d(2).to('gpu')
bn_gpu.train()
x_bn = forge.randn((2, 2, 2, 2), device='gpu')
out_bn_train = bn_gpu(x_bn)
await out_bn_train.realize()
await bn_gpu.running_mean.realize()
await bn_gpu.running_var.realize()
bn_gpu.eval()
out_bn_eval = bn_gpu(x_bn)
await out_bn_eval.realize()
bn_eval_passed = True

# 14. LayerNorm Parameter Identity Preservation:
ln_model = forge.nn.LayerNorm(4).to('gpu')
ln_opt = forge.optim.SGD(ln_model.parameters(), lr=0.1)
ln_weight_id_before = id(ln_model.weight)
x_ln = forge.randn((1, 4), device='gpu')
out_ln = ln_model(x_ln)
ln_param_identity_preserved = bool(id(ln_model.weight) == ln_weight_id_before and ln_opt.params[0] is ln_model.weight)

# 15. Verify p.grad handle disposal post step_async:
p_test = forge.tensor([1.0, 2.0], requires_grad=True).to('gpu')
p_test.grad = forge.tensor([0.1, 0.2]).to('gpu')
opt_test = forge.optim.SGD([p_test], lr=0.1)
await opt_test.step_async()
grad_disposed_and_cleared = bool(p_test.grad is None)

# 16. In-place parameter identity preserved when model.to('gpu') is called AFTER optimizer init:
cpu_linear = forge.nn.Linear(2, 4)
w_before = cpu_linear.weight
linear_opt = forge.optim.SGD(cpu_linear.parameters(), lr=0.1)
cpu_linear.to('gpu')
post_opt_to_gpu_identity = bool(cpu_linear.weight is w_before and linear_opt.params[0] is cpu_linear.weight and cpu_linear.weight.device == 'gpu')

# 17. PositionalEncoding dynamic seq_len execution on GPU:
pe_gpu = forge.nn.PositionalEncoding(d_model=8, max_len=50).to('gpu')
x_pe_1 = forge.zeros((1, 6, 8), device='gpu')
out_pe_1 = pe_gpu(x_pe_1)
await out_pe_1.realize()
x_pe_2 = forge.zeros((1, 10, 8), device='gpu')
out_pe_2 = pe_gpu(x_pe_2)
await out_pe_2.realize()
pe_dynamic_passed = bool(out_pe_1.shape == (1, 6, 8) and out_pe_2.shape == (1, 10, 8))

# 18. Model parameter GC finalizer verification post move_to_('gpu'):
import gc
temp_linear = forge.nn.Linear(2, 4)
temp_linear.to('gpu')
w_handle = temp_linear.weight._handle
del temp_linear
gc.collect()
forge.flush_gc(force=True)
handles_snapshot = forge.bridge.get_js_core().snapshotHandles().to_py()
gc_finalizer_passed = bool(w_handle not in handles_snapshot)

# 19. CrossEntropy on GPU with exact analytical gradient comparison:
ce_preds = forge.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], requires_grad=True).to('gpu')
ce_targets = forge.tensor([0, 1], device='cpu')
ce_loss = forge.functional.cross_entropy(ce_preds, ce_targets)
await ce_loss.realize()
ce_loss_val = await ce_loss.numpy_async()
ce_loss.backward()
await ce_preds.grad.realize()
ce_grad_val = await ce_preds.grad.numpy_async()

logits_np = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], dtype=np.float32)
exp_l = np.exp(logits_np - np.max(logits_np, axis=-1, keepdims=True))
probs_np = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
expected_grad_np = probs_np.copy()
expected_grad_np[0, 0] -= 1.0
expected_grad_np[1, 1] -= 1.0
expected_grad_np /= 2.0
ce_passed = bool(np.allclose(ce_grad_val, expected_grad_np, atol=1e-4))

# 19b. Sparse CrossEntropy backward with vector per-sample grad_output:
ce_preds_w = forge.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], requires_grad=True).to('gpu')
ce_targets_w = forge.tensor([0, 1], dtype='float32', device='gpu')
ce_grad_out = forge.tensor([2.0, 0.5], dtype='float32', device='gpu')
t_grad_logits = forge.Tensor(
    shape=ce_preds_w.shape,
    dtype='float32',
    device='gpu',
    op='sparse_cross_entropy_backward',
    parents=(ce_preds_w, ce_targets_w, ce_grad_out),
    op_params=[-100, 1.0]
)
await t_grad_logits.realize()
grad_w_val = await t_grad_logits.numpy_async()
expected_grad_w = probs_np.copy()
expected_grad_w[0, 0] -= 1.0
expected_grad_w[1, 1] -= 1.0
expected_grad_w[0] *= 2.0
expected_grad_w[1] *= 0.5
ce_vec_passed = bool(np.allclose(grad_w_val, expected_grad_w, atol=1e-4))

# 20. nn.Embedding on GPU with non-zero indices [2, 0, 1]:
emb_layer = forge.nn.Embedding(num_embeddings=4, embedding_dim=8)
emb_layer.to('gpu')
emb_tokens = forge.tensor([2, 0, 1], dtype='int32', device='gpu')
emb_out = emb_layer(emb_tokens)
await emb_out.realize()
emb_out_val = await emb_out.numpy_async()
emb_w_val = await emb_layer.weight.numpy_async()
emb_gpu_passed = bool(
    np.allclose(emb_out_val[0], emb_w_val[2]) and
    np.allclose(emb_out_val[1], emb_w_val[0]) and
    np.allclose(emb_out_val[2], emb_w_val[1])
)

{
    "bmm_diff": bmm_diff,
    "scat_diff": scat_diff,
    "gat_diff": gat_diff,
    "causal_diff": causal_diff,
    "sm_nan": sm_nan,
    "sm_sum_diff": sm_sum_diff,
    "max_axis_diff": max_axis_diff,
    "max_axis_backward_diff": max_axis_backward_diff,
    "broadcast_5d_diff": broadcast_5d_diff,
    "broadcast_6d_diff": broadcast_6d_diff,
    "broadcast_8d_diff": broadcast_8d_diff,
    "momentum_guard_passed": momentum_guard_passed,
    "strict_nan_caught": strict_nan_caught,
    "non_strict_nan_propagated": non_strict_nan_propagated,
    "bn_eval_passed": bn_eval_passed,
    "ln_param_identity_preserved": ln_param_identity_preserved,
    "grad_disposed_and_cleared": grad_disposed_and_cleared,
    "post_opt_to_gpu_identity": post_opt_to_gpu_identity,
    "pe_dynamic_passed": pe_dynamic_passed,
    "gc_finalizer_passed": gc_finalizer_passed,
    "ce_passed": ce_passed,
    "ce_vec_passed": ce_vec_passed,
    "emb_gpu_passed": emb_gpu_passed,
}
`;
    const res = await pyodide.runPythonAsync(pyCode);
    return res.toJs({ dict_converter: Object.fromEntries });
  });

  console.log('[Pyodide WebGPU Execution Verified]:', pyResult);
  expect(pyResult.bmm_diff).toBeLessThanOrEqual(ABS_TOL);
  expect(pyResult.scat_diff).toBeLessThanOrEqual(ABS_TOL);
  expect(pyResult.gat_diff).toBeLessThanOrEqual(ABS_TOL);
  expect(pyResult.causal_diff).toBeLessThanOrEqual(1e-3);
  expect(pyResult.sm_nan).toBe(false);
  expect(pyResult.sm_sum_diff).toBeLessThanOrEqual(1e-4);
  expect(pyResult.max_axis_diff).toBeLessThanOrEqual(ABS_TOL);
  expect(pyResult.max_axis_backward_diff).toBeLessThanOrEqual(ABS_TOL);
  expect(pyResult.broadcast_5d_diff).toBeLessThanOrEqual(ABS_TOL);
  expect(pyResult.broadcast_6d_diff).toBeLessThanOrEqual(ABS_TOL);
  expect(pyResult.broadcast_8d_diff).toBeLessThanOrEqual(ABS_TOL);
  expect(pyResult.momentum_guard_passed).toBe(true);
  expect(pyResult.strict_nan_caught).toBe(true);
  expect(pyResult.non_strict_nan_propagated).toBe(true);
  expect(pyResult.bn_eval_passed).toBe(true);
  expect(pyResult.ln_param_identity_preserved).toBe(true);
  expect(pyResult.grad_disposed_and_cleared).toBe(true);
  expect(pyResult.post_opt_to_gpu_identity).toBe(true);
  expect(pyResult.pe_dynamic_passed).toBe(true);
  expect(pyResult.gc_finalizer_passed).toBe(true);
  expect(pyResult.ce_passed).toBe(true);
  expect(pyResult.ce_vec_passed).toBe(true);
  expect(pyResult.emb_gpu_passed).toBe(true);
});
