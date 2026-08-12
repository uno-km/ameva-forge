export class AMEVAForgeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = new.target.name;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export class AMEVAForgeShapeError extends AMEVAForgeError {}
export class AMEVAForgeDTypeError extends AMEVAForgeError {}
export class AMEVAForgeDeviceError extends AMEVAForgeError {}
export class AMEVAForgeDisposedError extends AMEVAForgeError {}
export class AMEVAForgeQuotaExceededError extends AMEVAForgeError {}
export class AMEVAForgeWebGPUUnavailableError extends AMEVAForgeError {}
export class AMEVAForgeSecurityError extends AMEVAForgeError {}
