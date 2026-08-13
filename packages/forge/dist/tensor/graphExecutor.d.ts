/**
 * Created: 2026-08-12T12:14:52+09:00
 * Modified:
 *   - 2026-08-12T12:59:35+09:00: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
 *   - 2026-08-12T12:23:09+09:00: Docs: Build Apache-style docs and unify tests
 *   - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *
 * graphExecutor.ts — JSON 그래프 파서 & GPU 스케줄러
 *
 * C-04 Fix: JSON 입력에 대한 강력한 검증 추가
 * M-05 Fix: matmul dispatch X/Y swap 수정
 * H-01 Fix: _globalPipelineCache를 모든 op에 적용
 * NC-06 Fix: inst.in null-guard 추가 (! 비null 단언 제거)
 * NH-07 Fix: shaderGuard.assertAllowedKernelName() 실제 호출
 * NM-05 Fix: device.pushErrorScope()로 op별 에러 감지
 */
import { TensorHandle } from "../types";
/**
 * executeGraph — Python 레이지 그래프를 단일 FFI 호출로 GPU에 실행한다.
 * WHAT: Python 등 외부 환경에서 직렬화된 연산 그래프(JSON)를 받아 일괄적으로 GPU에서 실행하는 함수입니다.
 * WHY: 매 연산마다 JS와 WebAssembly/GPU 사이를 왕복(context switch)하면 극심한 오버헤드가 발생하므로, 한 번의 호출로 많은 명령을 처리(Transaction)하기 위해 설계되었습니다.
 * HOW: JSON을 파싱하고, 명령을 검증하며, 적절한 청크로 분할하여 WebGPU 커맨드 버퍼에 기록하고 제출(submit)합니다. 실패 시 트랜잭션을 롤백합니다.
 */
export declare function executeGraph(instructionsJson: string, jsInputs: unknown): Promise<Record<number, TensorHandle>>;
