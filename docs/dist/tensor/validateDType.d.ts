import { DType } from "../types";
/**
 * WHAT: 입력된 데이터 타입(dtype)이 프레임워크에서 지원하는 타입인지 검증하는 함수입니다.
 * WHY: 지원하지 않는 데이터 타입이 사용될 경우 발생할 수 있는 메모리 계산 오류 및 WebGPU 셰이더 오류를 사전에 방지하기 위해 존재합니다.
 * HOW: 입력된 dtype 문자열이 "float32"인지 비교하고, 일치하지 않으면 AMEVAForgeDTypeError 예외를 발생시킵니다. asserts 키워드를 사용하여 타입스크립트 컴파일러에게 dtype이 DType임을 보장합니다.
 */
export declare function validateDType(dtype: string): asserts dtype is DType;
