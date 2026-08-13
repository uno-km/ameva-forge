/**
 * Created: 2026-08-12 12:14:52 +0900
 * Modified:
 *   - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
 * 
 * types.ts — 핵심 타입 정의
 *
 * H-06 Fix: DType을 실제 지원되는 "float32"만으로 제한.
 *   기존에 float16/int32가 타입에 있었지만 셰이더와 검증 로직이 float32 전용이라
 *   타입 에러 없이 잘못된 셰이더에 전달되는 버그가 있었다.
 *   → float16/int32 추가는 셰이더 커널 구현과 동시에 이루어져야 한다.
 */

/**
 * WHAT: 텐서를 고유하게 식별하기 위한 핸들(Handle) 타입입니다.
 * WHY: 시스템 내부적으로 파이썬과 자바스크립트 간에 무겁고 복잡한 객체를 통째로 넘기지 않고, 문자열 형태의 참조값만 주고받음으로써 통신 오버헤드를 최소화하기 위해 고안되었습니다.
 * HOW: 타입스크립트의 타입 앨리어스(Type Alias)를 사용하여 string을 TensorHandle로 추상화 명명합니다.
 */
export type TensorHandle = string;

/** 
 * WHAT: 텐서의 데이터 타입을 정의하는 문자열 리터럴 타입입니다. (H-06: 현재 구현이 실제로 지원하는 dtype만 허용)
 * WHY: 현재 작성된 커스텀 셰이더와 연산 검증 로직이 32비트 부동소수점만 지원하므로, 잘못된 타입이 주입되어 런타임 버그나 셰이더 크래시가 발생하는 것을 컴파일 타임에 엄격히 제한하고 방지하기 위함입니다.
 * HOW: 타입스크립트의 리터럴 타입을 활용해 오직 "float32"라는 문자열 값만 허용하도록 고정합니다.
 */
export type DType = "float32";

import { AllocationToken } from "./webgpu/quota";

/**
 * WHAT: 메모리에 할당된 단일 텐서에 대한 전체 상태와 리소스 정보를 담고 있는 핵심 레코드 인터페이스입니다.
 * WHY: 텐서 레지스트리(Tensor Registry)가 텐서의 수명 주기(생성, 사용, 해제)를 완벽하게 추적하고 관리하기 위한 중앙 정보 저장소 역할을 제공하기 위해서 존재합니다.
 * HOW: GPUBuffer 리소스 자체와 차원(Shape), 데이터 타입(DType), 그리고 할당 토큰 등 다양한 메타데이터를 하나의 구조화된 객체 형태로 묶어서 표현합니다.
 */
export interface TensorRecord {
  /** WHAT: 텐서의 고유 식별자 문자열. WHY: 레지스트리에서 이 텐서를 특정하기 위해. HOW: UUID나 고유 해시 형태의 텍스트 저장. */
  handle: TensorHandle;
  
  /** WHAT: 텐서의 차원별 크기를 담은 숫자 배열. WHY: 다차원 데이터 구조(배치, 높이, 너비, 채널 등)를 해석하고 메모리 오프셋을 계산하기 위해. HOW: [정수, 정수, ...] 형태의 배열로 저장. */
  shape: number[];
  
  /** WHAT: 텐서 내 원소들의 데이터 타입. WHY: 데이터를 바이트로 변환하거나 셰이더에서 올바르게 읽을 수 있게 보장하기 위해. HOW: DType 타입(현재 "float32" 전용)으로 제한. */
  dtype: DType;
  
  /** WHAT: 버퍼가 차지하는 총 바이트 단위 크기. WHY: GPU 메모리 할당량을 계산하고 버퍼 복사 시 정확한 크기를 설정하기 위해. HOW: 숫자로 저장. */
  byteLength: number;
  
  /** WHAT: WebGPU 인스턴스의 실제 메모리 버퍼 객체 참조. WHY: 하드웨어 가속 연산 및 메모리 읽기/쓰기 작업을 직접 수행하기 위해. HOW: 네이티브 GPUBuffer 객체를 직접 레퍼런싱. */
  buffer: GPUBuffer;
  
  /** WHAT: QuotaManager로부터 발급받은 메모리 할당량을 추적하는 토큰. WHY: 시스템 전체 메모리 한도를 관리하고 자원 해제 시 할당량을 정확히 반납하기 위해. HOW: 발급된 토큰 객체를 참조로 가짐. */
  token: AllocationToken;
  
  /** WHAT: 해당 텐서의 메모리가 이미 해제(disposed)되었는지를 나타내는 플래그입니다. WHY: Use-After-Free 같은 비정상적인 메모리 접근을 사전에 차단하기 위해. HOW: 불리언(boolean) 값으로 상태를 저장하고 체크합니다. */
  disposed: boolean;
  
  /** 
   * WHAT: 단조 증가하는 등록 순서값입니다. (Monotonic registration order)
   * WHY: 타임스탬프의 경우 시스템 시간에 따라 값이 역행하거나 중복될 가능성이 있으므로, 순서 보장이 필요한 로직(예: 디버깅, LRU 캐싱 등)에서 완벽한 선후 관계를 판별하기 위해서입니다.
   * HOW: 내부 카운터를 증가시키며 얻은 고유 숫자를 할당합니다.
   */
  createdAt: number;

  /** WHAT: 트랜잭션 롤백 시 이 핸들에 매핑된 에러 메시지입니다. WHY: F-016 수정의 일환으로 개별 에러를 추적하기 위함. HOW: 문자열로 기록. */
  error?: string;
}

/**
 * WHAT: 외부에 텐서의 메타데이터(크기, 형태, 상태)만을 제공하기 위한 읽기 전용 형태의 인터페이스입니다.
 * WHY: 실제 GPUBuffer 객체나 내부 할당 토큰 같은 민감한 하드웨어 리소스를 숨기고, 파이썬(Pyodide) 쪽이나 외부에 텐서의 정보만 확인할 때 필요한 최소한의 데이터만 안전하게 노출하여 은닉성(Encapsulation)을 보장하기 위함입니다.
 * HOW: TensorRecord에서 민감한 속성을 제외한 서브셋(subset) 필드만으로 구성된 구조체를 정의합니다.
 */
export interface TensorInfo {
  /** WHAT: 텐서의 고유 식별자. WHY: 이 정보가 어떤 텐서의 것인지 매핑하기 위해. HOW: 식별자 복사. */
  handle: TensorHandle;
  /** WHAT: 텐서의 차원별 크기 배열. WHY: 외부에서 텐서의 모양을 파악하기 위해. HOW: 숫자 배열 반환. */
  shape: number[];
  /** WHAT: 텐서 데이터 타입. WHY: 외부에서 데이터 형식을 파악하기 위해. HOW: DType 값 반환. */
  dtype: DType;
  /** WHAT: 총 바이트 크기. WHY: 데이터 전송 크기 등을 예측하기 위해. HOW: 숫자 값 반환. */
  byteLength: number;
  /** WHAT: 해제(Disposed) 여부 확인 플래그. WHY: 외부에서 유효한 텐서인지 검사하기 위해. HOW: 불리언 값 반환. */
  disposed: boolean;
}
