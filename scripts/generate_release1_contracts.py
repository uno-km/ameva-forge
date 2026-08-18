#!/usr/bin/env python3
"""
generate_release1_contracts.py — Generates TypeScript & Python op schema contracts from single JSON source.
"""

import json
import os
import sys

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "../packages/forge/schema/release1-ops.json")
TS_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "../packages/forge/src/generated/opSchema.ts")
PY_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "../packages/forge-py/src/forge/generated/op_schema.py")
PY_INIT_PATH = os.path.join(os.path.dirname(__file__), "../packages/forge-py/src/forge/generated/__init__.py")


def generate_ts_code(schema: dict) -> str:
    ops = schema["operations"]
    ops_json = json.dumps(ops, indent=2)
    return f"""/**
 * AUTO-GENERATED FILE. DO NOT MODIFY MANUALLY.
 * Generated from packages/forge/schema/release1-ops.json
 * Run `py -3 scripts/generate_release1_contracts.py` to regenerate.
 */

export interface OpParamDef {{
  name: string;
  type: string;
}}

export interface OpDef {{
  inputs: number;
  params: OpParamDef[];
  output: string;
  dtypes: string[];
}}

export const RELEASE1_OP_SCHEMA: Record<string, OpDef> = {ops_json};

export type Release1OpName = keyof typeof RELEASE1_OP_SCHEMA;
"""


def generate_py_code(schema: dict) -> str:
    ops = schema["operations"]
    ops_repr = repr(ops)
    return f'''"""
AUTO-GENERATED FILE. DO NOT MODIFY MANUALLY.
Generated from packages/forge/schema/release1-ops.json
Run `py -3 scripts/generate_release1_contracts.py` to regenerate.
"""

from typing import Dict, Any, List

RELEASE1_OP_SCHEMA: Dict[str, Dict[str, Any]] = {ops_repr}

RELEASE1_OPS: List[str] = list(RELEASE1_OP_SCHEMA.keys())
'''


def main():
    check_mode = "--check" in sys.argv

    if not os.path.exists(SCHEMA_PATH):
        print(f"Error: Schema file not found at {SCHEMA_PATH}", file=sys.stderr)
        sys.exit(1)

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    ts_content = generate_ts_code(schema)
    py_content = generate_py_code(schema)

    if check_mode:
        mismatch = False
        if not os.path.exists(TS_OUTPUT_PATH):
            print(f"Drift Error: {TS_OUTPUT_PATH} does not exist.", file=sys.stderr)
            mismatch = True
        else:
            with open(TS_OUTPUT_PATH, "r", encoding="utf-8") as f:
                if f.read().strip() != ts_content.strip():
                    print(f"Drift Error: {TS_OUTPUT_PATH} is out of sync with schema.", file=sys.stderr)
                    mismatch = True

        if not os.path.exists(PY_OUTPUT_PATH):
            print(f"Drift Error: {PY_OUTPUT_PATH} does not exist.", file=sys.stderr)
            mismatch = True
        else:
            with open(PY_OUTPUT_PATH, "r", encoding="utf-8") as f:
                if f.read().strip() != py_content.strip():
                    print(f"Drift Error: {PY_OUTPUT_PATH} is out of sync with schema.", file=sys.stderr)
                    mismatch = True

        if mismatch:
            print("Run `py -3 scripts/generate_release1_contracts.py` to fix drift.", file=sys.stderr)
            sys.exit(1)
        else:
            print("Op contract check PASSED: All generated files are up to date.")
            sys.exit(0)
    else:
        os.makedirs(os.path.dirname(TS_OUTPUT_PATH), exist_ok=True)
        with open(TS_OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(ts_content)
        print(f"Generated {TS_OUTPUT_PATH}")

        os.makedirs(os.path.dirname(PY_OUTPUT_PATH), exist_ok=True)
        with open(PY_OUTPUT_PATH, "w", encoding="utf-8") as f:
            f.write(py_content)
        print(f"Generated {PY_OUTPUT_PATH}")

        with open(PY_INIT_PATH, "w", encoding="utf-8") as f:
            f.write('"""Generated contracts package."""\n')
        print(f"Generated {PY_INIT_PATH}")


if __name__ == "__main__":
    main()
