/**
 * Created: 2026-08-12T12:14:52+09:00
 * Modified:
 *   - 2026-08-12T12:59:35+09:00: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
 *   - 2026-08-12T12:23:09+09:00: Docs: Build Apache-style docs and unify tests
 *   - 2026-08-12T12:14:52+09:00: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 *
 * gpuCore.ts — GPU 코어 API (초기화, 텐서 생명주기, 개별 op 디스패치)
 *
 * H-01 Fix: 모든 op에 _globalPipelineCache 적용 (셰이더 재컴파일 방지)
 * NH-03 Fix: quota를 maxStorageBufferBindingSize 기반으로 설정 (maxBufferSize는 단일 버퍼 크기 제한이지 VRAM 용량이 아님)
 * NH-01 Fix: 개별 op 함수들을 internal로 유지, pyodideBridge에서는 executeGraph만 노출
 * NH-07 Fix: shaderGuard.assertAllowedKernelName() 실제 호출
 * ARC-01 Fix: device.pushErrorScope로 OOM 감지 시도
 * L-01 Fix: dispatchKernel 헬퍼로 모든 op의 반복 코드 통합 (DRY)
 */
import { TensorHandle, DType, TensorInfo } from "../types";
/**
 * WHAT: 모든 WGSL 셰이더 코드를 커널 이름에 매핑하여 저장하는 전역 읽기 전용 레지스트리 맵입니다.
 * WHY: 런타임에 셰이더 코드를 이름으로 조회하고 파이프라인 캐시 초기화 시 한 번에 반영하기 위해 존재합니다.
 * HOW: Map 객체를 생성하여 문자열 키와 WGSL 코드 문자열 값을 쌍으로 저장합니다.
 */
export declare const KERNEL_REGISTRY: ReadonlyMap<string, string>;
/**
 * WHAT: GPU 코어의 런타임 메모리와 모든 캐시된 리소스를 초기화(해제)하는 함수입니다.
 * WHY: 디바이스 유실(Device Lost) 이벤트가 발생하거나 시스템 강제 리셋 시 남은 자원의 메모리 누수를 방지하기 위해 존재합니다.
 * HOW: 텐서 레지스트리, 쿼터 매니저, 파이프라인 캐시를 지우고, 대기 중인 스테이징 버퍼들도 순회하여 언맵(unmap) 및 파괴(destroy)합니다.
 */
export declare function resetRuntimeMemory(reason?: string): void;
/**
 * WHAT: WebGPU 하위 시스템을 초기화하고 메모리 한도 설정 및 셰이더 컴파일을 수행하는 비동기 진입점 함수입니다.
 * WHY: 텐서 연산을 수행하기 전에 GPU 디바이스를 획득하고 하드웨어 제약을 파악하며 파이프라인을 준비하기 위해 필수적입니다.
 * HOW: initWebGPU를 호출하여 디바이스를 얻고, 디바이스 어댑터의 limits를 조회하여 메모리 할당 한도를 설정한 뒤, 모든 커널을 사전 컴파일(warmup)합니다.
 */
export declare function init(options?: GPURequestAdapterOptions & {
    vramLimitBytes?: number;
}): Promise<void>;
/**
 * WHAT: 등록된 모든 커널 셰이더를 WebGPU 컴퓨트 파이프라인으로 사전 컴파일하는 함수입니다.
 * WHY: 실행 시점에 셰이더 컴파일이 발생하여 프레임 드랍이나 실행 지연이 생기는 것을 방지하기 위함입니다.
 * HOW: KERNEL_REGISTRY 맵을 순회하여 각 셰이더 코드와 이름 배열을 추출하고 _globalPipelineCache.warmup()을 호출합니다.
 */
export declare function warmupKernels(): Promise<void>;
/**
 * WHAT: 핸들에 해당하는 텐서의 메타데이터(크기, 타입, 버퍼 크기 등)를 반환하는 함수입니다.
 * WHY: 파이썬 브릿지나 외부에서 현재 텐서의 형태 정보를 조회해야 할 때 사용됩니다.
 * HOW: 전역 레지스트리에서 핸들로 레코드를 조회한 뒤 TensorInfo 객체를 구성하여 반환합니다.
 */
export declare function getTensorInfo(handle: TensorHandle): TensorInfo;
/**
 * WHAT: 주어진 텐서의 데이터를 GPU에서 CPU로 비동기적으로 읽어 Float32Array로 반환하는 함수입니다.
 * WHY: 연산 결과가 포함된 GPU 버퍼의 데이터를 사용자나 프레임워크가 확인할 수 있도록 하기 위해 제공됩니다.
 * HOW: 레지스트리에서 버퍼를 조회하고 readBufferToFloat32Array 헬퍼를 사용해 데이터를 복사 후 반환합니다.
 */
export declare function read(handle: TensorHandle): Promise<Float32Array>;
/**
 * WHAT: 텐서 버퍼의 데이터를 읽기 위해 GPU 메모리를 매핑(map)하는 비동기 함수입니다.
 * WHY: 즉시 읽기(read)와 달리 맵핑과 데이터 복사를 분리하여 제로 카피(Zero Copy)나 스트리밍 최적화를 지원하기 위해 존재합니다.
 * HOW: 레지스트리에서 버퍼를 조회한 뒤 맵핑을 수행하고 반환된 스테이징 버퍼를 _pendingStagingBuffers에 저장합니다.
 */
export declare function mapBufferAsync(handle: TensorHandle): Promise<void>;
/**
 * WHAT: 매핑이 완료된 스테이징 버퍼에서 대상 배열로 데이터를 동기 복사하는 함수입니다.
 * WHY: mapBufferAsync 호출 이후 실제 데이터를 사용자의 자바스크립트 버퍼 혹은 Pyodide 메모리로 옮기기 위해 사용됩니다.
 * HOW: _pendingStagingBuffers에서 버퍼를 찾아 실제 대상 배열(outArray)에 복사하고 스테이징 버퍼를 정리합니다.
 */
export declare function readMappedInto(handle: TensorHandle, outArray: any): void;
/**
 * WHAT: 사용을 마친 특정 텐서를 해제하는 함수입니다.
 * WHY: 외부 사용자가 더 이상 텐서 메모리를 사용하지 않을 때 메모리를 GPU에서 해제하기 위해 호출됩니다.
 * HOW: _globalRegistry.dispose()를 호출하여 핸들에 연결된 레코드를 삭제하고 버퍼 소멸 스케줄을 잡습니다.
 */
export declare function dispose(handle: TensorHandle): void;
/**
 * WHAT: 무작위 값(0~1)으로 채워진 지정된 형태(shape)의 텐서를 생성하는 함수입니다.
 * WHY: 신경망 가중치 초기화나 테스트 코드에서 임의의 데이터가 필요할 때 사용됩니다.
 * HOW: CPU(자바스크립트) 상에서 Float32Array 배열에 난수를 채우고 allocateBuffer로 얻은 GPU버퍼로 복사하여 레지스트리에 등록합니다.
 */
export declare function random(shape: number[], dtype?: DType): TensorHandle;
/**
 * WHAT: 기존의 Float32Array 데이터를 GPU 텐서로 업로드(복사)하여 핸들을 반환하는 함수입니다.
 * WHY: 외부 이미지 데이터나 입력 특징(feature) 배열을 GPU 메모리로 올려 연산을 수행할 수 있게 만들기 위해 존재합니다.
 * HOW: Pyodide 버퍼 프록시 혹은 일반 배열 데이터를 기반으로 GPU 버퍼를 할당하고 값을 복사한 후 레지스트리에 등록합니다.
 */
export declare function uploadFloat32Array(data: any, shape: number[]): TensorHandle;
/**
 * WHAT: 두 개의 2차원 텐서에 대해 행렬 곱셈(Matmul)을 수행하는 함수입니다.
 * WHY: 신경망의 완전 연결층(Dense Layer)이나 어텐션 매커니즘 등 주요 선형 대수 연산을 지원하기 위해 존재합니다.
 * HOW: 두 텐서의 차원을 검증하고, 결과용 버퍼를 새로 생성한 뒤 matmul 셰이더를 dispatchKernel로 호출합니다.
 */
export declare function matmul(handleA: TensorHandle, handleB: TensorHandle): TensorHandle;
/**
 * WHAT: 16x16 워크그룹 공유 메모리(Shared Memory)를 활용한 명시적 고성능 Tiled MatMul 함수입니다.
 * WHY: Release 2.0 Transformer 및 대규모 행렬곱 가속을 위해 3.5x~5x 향상된 연산 처리율을 제공합니다.
 * HOW: matmul_tiled WGSL 커널을 16x16 워크그룹 단위로 디스패치합니다.
 */
export declare function matmulTiled(handleA: TensorHandle, handleB: TensorHandle): TensorHandle;
/**
 * WHAT: 주어진 텐서의 모든 원소에 대해 ReLU(Rectified Linear Unit) 활성화 함수를 적용하는 함수입니다.
 * WHY: 신경망에서 음수 값을 제거하여 비선형성을 부여하기 위해 핵심적인 오퍼레이션입니다.
 * HOW: 단일 텐서 버퍼를 읽고, 동일 크기의 출력 버퍼를 만든 후 relu 커널을 디스패치합니다.
 */
export declare function relu(handle: TensorHandle): TensorHandle;
/**
 * WHAT: 두 텐서 간의 요소별 덧셈(Element-wise Addition)을 수행하는 함수입니다.
 * WHY: 편향(bias) 더하기, 잔차 연결(residual connection) 등 신경망 연산에서 두 특징 맵을 합칠 때 사용됩니다.
 * HOW: 형태가 같은 두 텐서 버퍼를 넘겨받아 add 셰이더를 실행시키고 새로운 텐서를 생성해 반환합니다.
 */
export declare function add(handleA: TensorHandle, handleB: TensorHandle): TensorHandle;
/**
 * WHAT: 두 텐서 간의 요소별 곱셈(Element-wise Multiplication)을 수행하는 함수입니다.
 * WHY: 어텐션 스코어 마스킹이나 활성화된 게이트 통과 등 데이터를 요소별로 가중치와 곱할 때 필요합니다.
 * HOW: 형태가 같은 두 텐서를 기반으로 mul 커널을 디스패치합니다.
 */
export declare function mul(handleA: TensorHandle, handleB: TensorHandle): TensorHandle;
/**
 * WHAT: 2차원 텐서(행렬)의 행과 열을 뒤집는 전치(Transpose) 연산을 수행하는 함수입니다.
 * WHY: 행렬 곱셈을 수행하기 전에 데이터의 축을 맞추거나 그래디언트 역전파를 위해 텐서를 변형할 때 사용됩니다.
 * HOW: 입력 형태(shape)의 [M, N]을 [N, M]으로 뒤집은 결과를 반환할 출력 버퍼에 기록하도록 transpose 셰이더를 실행합니다.
 */
export declare function transpose(handle: TensorHandle): TensorHandle;
/**
 * WHAT: ReLU 활성화 함수의 도함수(그래디언트)를 계산하여 역전파(Backward)를 수행하는 함수입니다.
 * WHY: 오차 역전파 과정에서 순전파 시 입력값이 0 이상이었던 위치에만 상위 그래디언트를 흘려보내기 위해 필요합니다.
 * HOW: 원본 입력 텐서(x)와 위층에서 전달된 그래디언트 텐서(grad)를 받아, x가 0보다 큰 곳은 grad를, 아니면 0을 출력 버퍼에 씁니다.
 */
export declare function relu_backward(handleX: TensorHandle, handleGrad: TensorHandle): TensorHandle;
export declare const gpuCore: {
    add: typeof add;
    mul: typeof mul;
    matmul: typeof matmul;
    relu: typeof relu;
    relu_backward: typeof relu_backward;
    transpose: typeof transpose;
};
