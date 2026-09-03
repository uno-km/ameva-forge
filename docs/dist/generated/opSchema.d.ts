/**
 * AUTO-GENERATED FILE. DO NOT MODIFY MANUALLY.
 * Generated from packages/forge/schema/release1-ops.json
 * Run `py -3 scripts/generate_release1_contracts.py` to regenerate.
 */
export interface OpParamDef {
    name: string;
    type: string;
}
export interface OpDef {
    inputs: number;
    params: OpParamDef[];
    output: string;
    dtypes: string[];
}
export declare const RELEASE1_OP_SCHEMA: Record<string, OpDef>;
export type Release1OpName = keyof typeof RELEASE1_OP_SCHEMA;
