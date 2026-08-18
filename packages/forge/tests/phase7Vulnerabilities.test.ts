import { executeGraph } from "../src/tensor/graphExecutor";
import { _stagingPool, clearStagingPool, acquireStagingBuffer, releaseStagingBuffer, allocateBuffer, freeBuffer } from "../src/webgpu/buffers";
import { _setDeviceForTesting } from "../src/webgpu/device";
import { AMEVAForgeValidationError, AMEVAForgeSecurityError } from "../src/errors";
import { SUM_WGSL } from "../src/tensor/kernels/sum.wgsl";
import { MAX_WGSL } from "../src/tensor/kernels/max.wgsl";
import { SUM_AXIS_WGSL } from "../src/tensor/kernels/sum_axis.wgsl";
import { ADD_WGSL } from "../src/tensor/kernels/add.wgsl";

describe("Phase 7 Deep Vulnerabilities Unit Test Suite (SCRUM-153 ~ SCRUM-164)", () => {
  const mockDevice: any = {
    createBuffer: jest.fn(() => ({ destroy: jest.fn() })),
    createCommandEncoder: jest.fn(() => ({
      beginComputePass: jest.fn(() => ({
        setPipeline: jest.fn(),
        setBindGroup: jest.fn(),
        dispatchWorkgroups: jest.fn(),
        end: jest.fn()
      })),
      copyBufferToBuffer: jest.fn(),
      finish: jest.fn()
    })),
    createBindGroup: jest.fn(() => ({})),
    createShaderModule: jest.fn(() => ({})),
    createComputePipeline: jest.fn(() => ({
      getBindGroupLayout: jest.fn(() => ({}))
    })),
    pushErrorScope: jest.fn(),
    popErrorScope: jest.fn().mockResolvedValue(null),
    queue: {
      writeBuffer: jest.fn(),
      submit: jest.fn(),
      onSubmittedWorkDone: jest.fn().mockResolvedValue(undefined)
    }
  };

  beforeEach(() => {
    _setDeviceForTesting(mockDevice);
    clearStagingPool();
  });

  afterAll(() => {
    _setDeviceForTesting(null);
  });

  describe("SCRUM-153 (VULN-01): 8D Stride Broadcasting Shader Contract", () => {
    it("should have 8D coordinate decoder in ADD_WGSL", () => {
      expect(ADD_WGSL).toContain("stride_a0: u32");
      expect(ADD_WGSL).toContain("stride_a7: u32");
      expect(ADD_WGSL).toContain("stride_b0: u32");
      expect(ADD_WGSL).toContain("stride_b7: u32");
      expect(ADD_WGSL).toContain("params.stride_a7");
      expect(ADD_WGSL).toContain("params.stride_b7");
    });
  });

  describe("SCRUM-154 (VULN-02): Generic sum_axis 3D/4D Reduction Shader Contract", () => {
    it("should use outer_size, reduction_size, inner_stride in SUM_AXIS_WGSL", () => {
      expect(SUM_AXIS_WGSL).toContain("outer_size: u32");
      expect(SUM_AXIS_WGSL).toContain("reduction_size: u32");
      expect(SUM_AXIS_WGSL).toContain("inner_stride: u32");
      expect(SUM_AXIS_WGSL).toContain("let outer_idx = out_idx / inner_stride;");
      expect(SUM_AXIS_WGSL).toContain("let inner_idx = out_idx % inner_stride;");
    });
  });

  describe("SCRUM-155 & SCRUM-156 (VULN-03 & VULN-04): StagingBufferPool Token Ownership & Device Lost Clear", () => {
    it("should manage staging buffers with tokens and clear them completely on clearStagingPool", () => {
      expect(_stagingPool.size).toBe(0);
      const { buffer, token } = acquireStagingBuffer(1024);
      expect(buffer).toBeDefined();
      expect(token).toBeDefined();

      releaseStagingBuffer(buffer, token, 1024);
      expect(_stagingPool.size).toBe(1);
      const entries = _stagingPool.get(1024);
      expect(entries).toBeDefined();
      expect(entries![0].token).toBe(token);

      clearStagingPool();
      expect(_stagingPool.size).toBe(0);
    });
  });

  describe("SCRUM-157 (VULN-05): Reduction Linear Workgroup Decoder for >65535 Workgroups", () => {
    it("should restore linear workgroup id from 2D dispatch in SUM_WGSL and MAX_WGSL", () => {
      expect(SUM_WGSL).toContain("let wg_linear = workgroup_id.y * params.workgroups_x + workgroup_id.x;");
      expect(SUM_WGSL).toContain("output[wid] = s_data[0];");
      expect(MAX_WGSL).toContain("let wg_linear = workgroup_id.y * params.workgroups_x + workgroup_id.x;");
    });
  });

  describe("SCRUM-158 (VULN-06): AXPY In-Place Commit Phase Op Isolation", () => {
    it("should reject any downstream non-axpy operations following an AXPY instruction", async () => {
      const invalidGraph = JSON.stringify([
        { id: 1, op: "upload", shape: [4], dtype: "float32" },
        { id: 2, op: "upload", shape: [4], dtype: "float32" },
        { id: 3, op: "axpy", shape: [4], dtype: "float32", in: [1, 2], params: [4, 0.01] },
        { id: 4, op: "relu", shape: [4], dtype: "float32", in: [2] },
      ]);

      const in0 = new Float32Array([1, 2, 3, 4]);
      const in1 = new Float32Array([5, 6, 7, 8]);

      await expect(executeGraph(invalidGraph, [in0, in1])).rejects.toThrow(AMEVAForgeSecurityError);
    });
  });

  describe("SCRUM-162 (VULN-10): Fail-Fast NaN/Inf Upload Validation", () => {
    it("should throw AMEVAForgeValidationError when uploading NaN without allowNonFinite flag", async () => {
      const graph = JSON.stringify([
        { id: 1, op: "upload", shape: [4], dtype: "float32" }
      ]);
      const nanData = new Float32Array([1.0, NaN, 3.0, 4.0]);

      await expect(executeGraph(graph, [nanData])).rejects.toThrow(AMEVAForgeValidationError);
    });

    it("should throw AMEVAForgeValidationError when uploading Infinity without allowNonFinite flag", async () => {
      const graph = JSON.stringify([
        { id: 1, op: "upload", shape: [4], dtype: "float32" }
      ]);
      const infData = new Float32Array([1.0, Infinity, 3.0, 4.0]);

      await expect(executeGraph(graph, [infData])).rejects.toThrow(AMEVAForgeValidationError);
    });

    it("should ignore allowNonFinite in untrusted JSON graph instruction and still reject NaN", async () => {
      const maliciousGraph = JSON.stringify([
        { id: 1, op: "upload", shape: [4], dtype: "float32", allowNonFinite: true }
      ]);
      const nanData = new Float32Array([1.0, NaN, 3.0, 4.0]);

      await expect(executeGraph(maliciousGraph, [nanData])).rejects.toThrow(AMEVAForgeValidationError);
    });
  });

  describe("Audit VULN-01: Reshape Op Support", () => {
    it("should successfully execute reshape operation in graph transaction", async () => {
      const graph = JSON.stringify([
        { id: 1, op: "upload", shape: [2, 2], dtype: "float32" },
        { id: 2, op: "reshape", shape: [4], dtype: "float32", in: [1] }
      ]);
      const data = new Float32Array([1.0, 2.0, 3.0, 4.0]);
      const res = await executeGraph(graph, [data]);
      expect(res[2]).toBeDefined();
      expect(res[2]).toMatch(/^tensor_/);
    });
  });

  describe("Audit VULN-03: Corrupted Staging Buffer Purge", () => {
    it("should purge corrupted staging buffers instead of recycling them into pool", () => {
      expect(_stagingPool.size).toBe(0);
      const { buffer, token } = acquireStagingBuffer(512);
      releaseStagingBuffer(buffer, token, 512, true); // corrupted
      expect(_stagingPool.size).toBe(0);
    });
  });

  describe("Audit 2nd Pass: Chained Permute Op Input Resolution", () => {
    it("should correctly resolve inShape for permute when input is an intermediate node in same graph", async () => {
      const graph = JSON.stringify([
        { id: 1, op: "upload", shape: [2, 3], dtype: "float32" },
        { id: 2, op: "relu", shape: [2, 3], dtype: "float32", in: [1] },
        { id: 3, op: "permute", shape: [3, 2], dtype: "float32", in: [2], params: [1, 0] }
      ]);
      const data = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
      const res = await executeGraph(graph, [data]);
      expect(res[3]).toBeDefined();
      expect(res[3]).toMatch(/^tensor_/);
    });
  });

  describe("Audit 3rd Pass: Large Tensor 2D Dispatch (>4M elements)", () => {
    it("should split large fill (>4,194,240 elements) into 2D workgroups", async () => {
      const largeNumel = 5_000_000;
      const graph = JSON.stringify([
        { id: 1, op: "fill", shape: [largeNumel], dtype: "float32", params: [largeNumel, 7.0] }
      ]);
      const res = await executeGraph(graph, []);
      expect(res[1]).toBeDefined();
      expect(res[1]).toMatch(/^tensor_/);
    });

    it("should split large pad into 2D workgroups", async () => {
      const graph = JSON.stringify([
        { id: 1, op: "upload", shape: [1000, 5000], dtype: "float32" },
        {
          id: 2,
          op: "pad",
          shape: [1002, 5002],
          dtype: "float32",
          in: [1],
          params: [
            1002 * 5002, // num_elements (>5M)
            2,           // rank
            0.0,         // pad_val
            0,           // workgroups_x placeholder
            5000, 1, 0, 0, 0, 0, 0, 0, // in_strides
            5002, 1, 0, 0, 0, 0, 0, 0, // out_strides
            1, 1, 0, 0, 0, 0, 0, 0,    // pad_before
            1000, 5000, 0, 0, 0, 0, 0, 0 // in_shape
          ]
        }
      ]);
      const data = new Float32Array(1000 * 5000);
      const res = await executeGraph(graph, [data]);
      expect(res[2]).toBeDefined();
      expect(res[2]).toMatch(/^tensor_/);
    });
  });

  describe("P0-1: Direct TS API (gpuCore.ts) 112-Byte 8D Layout Contract", () => {
    it("should construct 112-byte (28 uint32) uniform buffer when calling direct gpuCore.add", () => {
      const { add } = require("../src/tensor/gpuCore");
      const { _globalRegistry } = require("../src/tensor/tensorRegistry");
      const mockBufA = { destroy: jest.fn() };
      const mockBufB = { destroy: jest.fn() };
      const hA = _globalRegistry.register({ buffer: mockBufA, token: "tok1", shape: [2, 3], dtype: "float32", byteLength: 24 });
      const hB = _globalRegistry.register({ buffer: mockBufB, token: "tok2", shape: [2, 3], dtype: "float32", byteLength: 24 });

      const outH = add(hA, hB);
      expect(outH).toBeDefined();
      expect(outH).toMatch(/^tensor_/);
      
      // Verify writeBuffer was called with 112-byte paramsData (28 uint32 elements)
      const writeCalls = mockDevice.queue.writeBuffer.mock.calls;
      const lastCall = writeCalls[writeCalls.length - 1];
      const paramsArray = lastCall[2];
      expect(paramsArray.byteLength).toBe(112);
      expect(paramsArray.length).toBe(28);
      expect(paramsArray[0]).toBe(6); // numElements = 2 * 3
      expect(paramsArray[2]).toBe(2); // rank = 2
    });
  });
});
