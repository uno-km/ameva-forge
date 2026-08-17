/**
 * errorPropagation.test.ts ??Release 1 error type verification
 *
 * Validates that:
 * 1. All required error types exist and extend AMEVAForgeError
 * 2. Error types are properly exported
 * 3. Error instances have correct names
 */

import {
  AMEVAForgeError,
  AMEVAForgeShapeError,
  AMEVAForgeDTypeError,
  AMEVAForgeDeviceError,
  AMEVAForgeDisposedError,
  AMEVAForgeQuotaExceededError,
  AMEVAForgeWebGPUUnavailableError,
  AMEVAForgeSecurityError,
  AMEVAForgeUnsupportedOpError,
  AMEVAForgeValidationError,
  AMEVAForgeOutOfMemoryError,
  AMEVAForgeInternalGPUError,
  AMEVAForgeDeviceLostError,
  AMEVAForgeStaleHandleError,
} from '../src/errors';

describe('Error Type Hierarchy', () => {
  const errorClasses = [
    { cls: AMEVAForgeShapeError, name: 'AMEVAForgeShapeError' },
    { cls: AMEVAForgeDTypeError, name: 'AMEVAForgeDTypeError' },
    { cls: AMEVAForgeDeviceError, name: 'AMEVAForgeDeviceError' },
    { cls: AMEVAForgeDisposedError, name: 'AMEVAForgeDisposedError' },
    { cls: AMEVAForgeQuotaExceededError, name: 'AMEVAForgeQuotaExceededError' },
    { cls: AMEVAForgeWebGPUUnavailableError, name: 'AMEVAForgeWebGPUUnavailableError' },
    { cls: AMEVAForgeSecurityError, name: 'AMEVAForgeSecurityError' },
    { cls: AMEVAForgeUnsupportedOpError, name: 'AMEVAForgeUnsupportedOpError' },
    // Release 1 new error types
    { cls: AMEVAForgeValidationError, name: 'AMEVAForgeValidationError' },
    { cls: AMEVAForgeOutOfMemoryError, name: 'AMEVAForgeOutOfMemoryError' },
    { cls: AMEVAForgeInternalGPUError, name: 'AMEVAForgeInternalGPUError' },
    { cls: AMEVAForgeDeviceLostError, name: 'AMEVAForgeDeviceLostError' },
    { cls: AMEVAForgeStaleHandleError, name: 'AMEVAForgeStaleHandleError' },
  ];

  it.each(errorClasses)('$name extends AMEVAForgeError', ({ cls, name }) => {
    const instance = new cls(`test ${name}`);
    expect(instance).toBeInstanceOf(AMEVAForgeError);
    expect(instance).toBeInstanceOf(Error);
    expect(instance.name).toBe(name);
    expect(instance.message).toBe(`test ${name}`);
  });

  it('AMEVAForgeError extends Error', () => {
    const err = new AMEVAForgeError('base error');
    expect(err).toBeInstanceOf(Error);
    expect(err.name).toBe('AMEVAForgeError');
    expect(err.stack).toBeDefined();
  });

  describe('GPU error scope types', () => {
    it('AMEVAForgeValidationError preserves message', () => {
      const err = new AMEVAForgeValidationError('GPU Validation Error: invalid usage');
      expect(err.message).toContain('GPU Validation Error');
    });

    it('AMEVAForgeOutOfMemoryError preserves message', () => {
      const err = new AMEVAForgeOutOfMemoryError('GPU Out of Memory: allocation failed');
      expect(err.message).toContain('Out of Memory');
    });

    it('AMEVAForgeInternalGPUError preserves message', () => {
      const err = new AMEVAForgeInternalGPUError('Internal GPU Error: shader compilation');
      expect(err.message).toContain('Internal GPU Error');
    });
  });
});
