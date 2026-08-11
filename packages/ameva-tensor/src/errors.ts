export class AMEVATensorError extends Error {
  constructor(message: string) {
    super(message);
    this.name = new.target.name;
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export class AMEVATensorShapeError extends AMEVATensorError {}
export class AMEVATensorDTypeError extends AMEVATensorError {}
export class AMEVATensorDeviceError extends AMEVATensorError {}
export class AMEVATensorDisposedError extends AMEVATensorError {}
export class AMEVATensorQuotaExceededError extends AMEVATensorError {}
export class AMEVATensorWebGPUUnavailableError extends AMEVATensorError {}
export class AMEVATensorSecurityError extends AMEVATensorError {}
