import * as forge from '../src/index';

describe('Public Security & API Boundary Contract', () => {
  it('does not export raw GPUDevice or getDevice directly on public surface', () => {
    expect((forge as any).getDevice).toBeUndefined();
    expect((forge as any).getAdapter).toBeUndefined();
    expect((forge as any).getQueue).toBeUndefined();
  });

  it('exposes safe lifecycle and management APIs', () => {
    expect(typeof forge.initWebGPU).toBe('function');
    expect(typeof forge.isAvailable).toBe('function');
    expect(typeof forge.executeGraph).toBe('function');
    expect(typeof forge.getQuotaSnapshot).toBe('function');
    expect(typeof forge.uploadFloat32Array).toBe('function');
    expect(typeof forge.mapBufferAsync).toBe('function');
    expect(typeof forge.readMappedInto).toBe('function');
    expect(typeof forge.dispose).toBe('function');
  });

  it('attaches __testing hooks only in test mode', () => {
    expect(forge.__testing).toBeDefined();
    expect(typeof forge.__testing?.destroyDevice).toBe('function');
    expect(typeof forge.__testing?.triggerValidationError).toBe('function');
    expect(typeof forge.__testing?.setQuotaLimit).toBe('function');
  });

  it('verifies registerKernelNames is additive and protects core whitelist', () => {
    const originalNames = forge.getAllowedKernelNames();
    expect(originalNames.has('matmul')).toBe(true);
    expect(originalNames.has('relu')).toBe(true);
    
    // Test additive registration
    forge.registerKernelNames(['custom_kernel_v1']);
    expect(forge.getAllowedKernelNames().has('custom_kernel_v1')).toBe(true);
    expect(forge.getAllowedKernelNames().has('matmul')).toBe(true); // Base kernels intact

    // Test rejection of malicious identifier
    expect(() => {
      forge.registerKernelNames(['malicious; drop table;']);
    }).toThrow();
  });
});
