export declare class AMEVATensorError extends Error {
    constructor(message: string);
}
export declare class AMEVATensorShapeError extends AMEVATensorError {
}
export declare class AMEVATensorDTypeError extends AMEVATensorError {
}
export declare class AMEVATensorDeviceError extends AMEVATensorError {
}
export declare class AMEVATensorDisposedError extends AMEVATensorError {
}
export declare class AMEVATensorQuotaExceededError extends AMEVATensorError {
}
export declare class AMEVATensorWebGPUUnavailableError extends AMEVATensorError {
}
export declare class AMEVATensorSecurityError extends AMEVATensorError {
}
