"""
=============================================================================
[파일 이력 메타데이터]
- Created: 2026-08-12 12:14:52 +0900
- Modified:
  - 2026-08-12 12:14:52 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
=============================================================================
데이터 로딩 및 미니배치 생성을 담당하는 유틸리티 모듈입니다.
"""
import numpy as np
from .ops import tensor
from .tensor import Tensor


class DataLoader:
    """
    [WHAT] 
    모델 학습에 사용할 데이터를 일정한 크기의 미니배치(mini-batch)로 쪼개어 제공하는 이터레이터 클래스입니다.
    
    [WHY] 
    전체 데이터를 한 번에 GPU 메모리에 올리면 OOM이 발생할 수 있으므로, 설정한 batch_size만큼 분할하여 효율적인 학습 반복을 수행하기 위해 필요합니다.
    
    [HOW] 
    생성자에서 입력과 라벨 데이터를 저장하고 인덱스를 셔플링할지 결정하며, __iter__ 메서드를 통해 미니배치 분량만큼의 데이터를 Tensor 객체로 래핑해 반환합니다.
    
    VUL-010 Fix: label_dtype 파라미터 추가.
      - 'float32' (기본): regression/one-hot target용
      - 'int64': classification 정수 라벨 유지
      - 'auto': y_data의 원본 dtype이 정수형이면 int64, 아니면 float32
    """
    def __init__(self, x_data, y_data, batch_size=32, shuffle=True, label_dtype='auto'):
        """
        [WHAT] 
        DataLoader 클래스의 인스턴스를 초기화하는 생성자입니다.
        
        [WHY] 
        데이터셋 원본을 클래스 내부에 보관하고, 미니배치 크기나 셔플 여부, 라벨의 데이터 타입 같은 동작 설정값들을 인스턴스에 저장하기 위함입니다.
        
        [HOW] 
        입력 특성인 x_data를 float32 numpy 배열로 변환하고, y_data의 경우 label_dtype 옵션에 맞춰 적절한 numpy 자료형을 선택한 후 타입 변환을 수행하여 인스턴스 변수에 할당합니다.
        """
        # 입력 데이터 x_data를 가져와 모든 연산의 기본인 float32 타입의 numpy 배열로 강제 변환하여 저장합니다.
        self.x_data = np.array(x_data, dtype=np.float32)

        # VUL-010: 라벨 dtype 보존
        # 라벨 데이터 y_data도 우선 기본적인 numpy 배열로 캐스팅합니다.
        y_arr = np.array(y_data)
        
        if label_dtype == 'auto':
            # 사용자가 auto로 설정한 경우 원본 배열의 데이터 타입을 확인합니다.
            if np.issubdtype(y_arr.dtype, np.integer):
                # 원본 타입이 정수 계열이라면 손실 없이 분류 문제를 풀기 위해 int64 타입을 선택합니다.
                self._label_dtype = np.int64
            else:
                # 정수형이 아니라면 기본적으로 회귀 문제나 원-핫 인코딩으로 간주해 float32 타입을 선택합니다.
                self._label_dtype = np.float32
        elif label_dtype == 'int64':
            # 사용자가 명시적으로 int64를 요구했다면 해당 타입을 선택합니다.
            self._label_dtype = np.int64
        else:
            # 그 외의 모든 경우(주로 float32가 들어옴)에는 기본적으로 float32 타입을 선택합니다.
            self._label_dtype = np.float32

        # 위에서 결정된 타입(_label_dtype)을 적용하여 y_data를 안전하게 형변환한 최종 라벨 배열을 저장합니다.
        self.y_data = y_arr.astype(self._label_dtype)
        # 미니배치 한 개당 몇 개의 샘플을 포함할지 결정하는 batch_size 크기를 저장합니다.
        self.batch_size = batch_size
        # 에포크마다 데이터를 무작위로 섞을지 여부를 결정하는 boolean 플래그를 저장합니다.
        self.shuffle = shuffle

    def __iter__(self):
        """
        [WHAT] 
        클래스 객체를 반복 가능(iterable)하게 만들어주는 매직 메서드입니다.
        
        [WHY] 
        파이썬의 for 루프에서 이 로더를 바로 순회하면서 미니배치 쌍(x_batch, y_batch)을 하나씩 꺼내어 학습 루프에 공급해야 하기 때문입니다.
        
        [HOW] 
        데이터의 총 개수만큼 인덱스 배열을 생성하고(옵션에 따라 셔플 적용), batch_size만큼 슬라이싱한 인덱스를 이용해 데이터를 추출, Tensor 형태로 래핑하여 차례대로 yield 합니다.
        """
        # 전체 데이터 셋의 총 샘플 개수(n)를 구합니다.
        n = len(self.x_data)
        # 0부터 n-1까지의 연속된 정수 인덱스 배열을 생성합니다.
        indices = np.arange(n)
        
        if self.shuffle:
            # 셔플 플래그가 True라면 인덱스 배열을 무작위 순서로 섞어 배치 구성이 랜덤하게 이루어지게 합니다.
            np.random.shuffle(indices)
            
        # 0부터 n까지 batch_size만큼의 간격(step)으로 루프를 돌며 배치의 시작 인덱스(start)를 잡습니다.
        for start in range(0, n, self.batch_size):
            # 시작 인덱스에서 batch_size를 더하여 배치의 끝 인덱스(end)를 구하되, 총 개수 n을 초과하지 않도록 보정합니다.
            end = min(start + self.batch_size, n)
            # 설정된 범위(start:end)만큼 인덱스 배열을 슬라이싱하여 현재 배치의 데이터 인덱스 묶음을 만듭니다.
            batch_idx = indices[start:end]
            # 해당 인덱스에 매칭되는 입력 특성 데이터를 추출하고 Tensor 객체로 래핑하여 텐서 그래프에 연결할 준비를 합니다.
            x_batch = tensor(self.x_data[batch_idx])
            # 라벨은 원본 dtype 유지하여 Tensor 생성
            # 해당 인덱스에 매칭되는 라벨 원본 데이터를 추출합니다.
            y_batch_data = self.y_data[batch_idx]
            
            if self._label_dtype == np.int64:
                # 정수 라벨: float32 Tensor로 변환하되 값은 정수로 유지
                # cross_entropy 등에서 .astype(np.int64)로 복원 가능
                # 현재 엔진이 float32 텐서 구조를 가정하므로 우선 float32로 캐스팅하여 Tensor 객체를 생성합니다.
                y_batch = tensor(y_batch_data.astype(np.float32))
            else:
                # 라벨 타입이 float32 계열이라면 별도의 변환 없이 곧바로 Tensor 객체를 생성합니다.
                y_batch = tensor(y_batch_data)
                
            # 구성이 완료된 미니배치 쌍(x_batch, y_batch)을 호출자(학습 루프 등)에게 반환하고 실행 상태를 일시정지합니다.
            yield x_batch, y_batch

    def __len__(self):
        """
        [WHAT] 
        전체 데이터셋을 처리할 때 발생할 수 있는 전체 미니배치(스텝)의 개수를 반환합니다.
        
        [WHY] 
        진행률(progress bar) 표시, 스텝 단위 스케줄러 업데이트 등 학습 중에 총 반복 횟수를 사전에 알아야 할 때가 많기 때문입니다.
        
        [HOW] 
        전체 데이터 크기에서 batch_size - 1을 더한 값을 batch_size로 나누는 올림 나눗셈을 통해 총 미니배치 수를 정수로 계산하여 반환합니다.
        """
        # 총 데이터 갯수를 배치 사이즈로 나눈 후 남은 나머지 데이터도 한 배치를 구성하게 되므로 이 식을 이용해 총 배치 수를 도출하여 반환합니다.
        return (len(self.x_data) + self.batch_size - 1) // self.batch_size

