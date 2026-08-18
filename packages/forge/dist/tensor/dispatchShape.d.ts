export type Dispatch2D = {
    dispatchX: number;
    dispatchY: number;
    workgroupsX: number;
    totalWorkgroups: number;
};
export declare function computeDispatch2D(numElements: number, workgroupSize?: number, maxPerDim?: number): Dispatch2D;
