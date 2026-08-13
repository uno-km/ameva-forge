# Python, TypeScript, WGSL Compatibility Matrix

| API / Op | Python GPU Lazy Node | TypeScript ALLOWED_OPS | WGSL Kernel | Readback | Maturity | Release 1 Decision |
|---|---|---|---|---|---|---|
| `tensor` / `upload` | Yes (`upload`) | Yes | N/A (Buffer Write) | Yes | Verified Beta | Include |
| `add`, `sub`, `mul`, `div`, `neg` | Yes | Yes | Yes | Yes | Verified Beta | Include |
| `matmul` (2D) | Yes | Yes | Yes | Yes | Verified Beta | Include |
| `batched_matmul` | Yes | Yes | Yes | Yes | Partial | Experimental |
| `relu`, `sigmoid`, `tanh` | Yes | Yes | Yes | Yes | Verified Beta | Include |
| `exp`, `log` | Yes | Yes | Yes | Yes | Partial | Experimental |
| `sum`, `max` (Scalar) | Yes | Yes | Yes (Buggy) | Buggy (Returns Partial) | **Broken** | Include (Requires Fix) |
| `mean` | Yes (sum/count) | N/A | N/A | N/A | **Broken** (due to sum) | Include (Requires Fix) |
| `reshape` | Yes (`reshape`) | **No** | N/A | N/A | **Broken** | Include (Requires Fix) |
| `unsqueeze`, `squeeze`, `flatten` | Yes (via `reshape`) | **No** | N/A | N/A | **Broken** | Experimental |
| `transpose` (2D) | Yes | Yes | Yes | Yes | Partial | Include |
| `permute` (ND) | Yes | Yes | Yes | Yes | Not Verified | Experimental |
| `Conv2d`, `maxpool2d` | Yes | Yes | Yes | Yes | Not Verified | **Exclude** |

*Note: All API items are considered "Verified Beta" at best since browser E2E test suite does not exist. "Partial" means backward parity or exact bounds aren't fully tested.*
