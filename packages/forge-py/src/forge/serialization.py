"""
================================================================================
파일 이력 (Historical Metadata)
Created: 2026-08-12 12:59:35 +0900 (첫 커밋 기준)
Modified:
  - 2026-08-12 12:59:35 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
================================================================================
이 파일은 모델의 가중치(상태)를 파일 시스템에 직렬화하여 저장하고, 
저장된 상태를 다시 메모리로 불러오는 기능을 제공합니다.
"""
# numpy 배열을 다루기 위한 라이브러리 임포트 (데이터 직렬화 및 역직렬화에 사용)
import numpy as np
# 신경망 모델의 기본 클래스 임포트
from .nn import Module

def save_model(model: Module, path: str):
    """
    WHAT: 주어진 모델의 가중치를 파일로 저장하는 함수입니다.
    WHY: 학습이 완료된 모델을 나중에 다시 사용하거나 배포하기 위해 파일 시스템에 상태를 영구적으로 보존할 필요가 있기 때문입니다.
    HOW: 모델의 state_dict를 호출하여 파라미터 딕셔너리를 얻고, 이를 numpy 포맷(.npz)으로 변환하여 저장합니다.
    """
    # WHAT: 모델의 현재 상태(가중치 등)를 담은 딕셔너리입니다.
    # WHY: 파일로 직렬화할 데이터를 추출하기 위함입니다.
    # HOW: keep_vars=False로 설정하여 순수 데이터만 추출합니다.
    state_dict = model.state_dict(keep_vars=False)
    
    # WHAT: numpy 배열 형태의 데이터를 담을 빈 딕셔너리입니다.
    # WHY: np.savez 함수에 전달하기 위해서는 모든 값이 numpy 배열이어야 하기 때문입니다.
    # HOW: 빈 딕셔너리로 초기화한 후 반복문을 통해 값을 채웁니다.
    numpy_dict = {}
    
    # WHAT: 모델의 상태 딕셔너리 내 모든 키(k)와 값(v) 쌍을 순회하는 반복문입니다.
    # WHY: 텐서 형태의 값들을 numpy 배열로 변환하여 numpy_dict에 옮겨 담기 위함입니다.
    # HOW: items() 메서드를 호출하여 반환된 키-값 쌍에 대해 반복합니다.
    for k, v in state_dict.items():
        if hasattr(v, 'numpy'):
            # WHAT: 텐서를 numpy 배열로 변환하여 저장합니다.
            # WHY: Tensor 객체는 직접 저장할 수 없으므로 호환 가능한 형식으로 변환해야 합니다.
            # HOW: v.numpy()를 호출한 값을 키 k로 numpy_dict에 할당합니다.
            numpy_dict[k] = v.numpy()
        else:
            # WHAT: 텐서가 아닌 값을 그대로 저장합니다.
            # WHY: 이미 numpy 배열이거나 기본 자료형인 경우 추가 변환이 필요 없기 때문입니다.
            # HOW: 원본 값을 그대로 키 k로 numpy_dict에 할당합니다.
            numpy_dict[k] = v
            
    # WHAT: 딕셔너리의 내용을 파일로 저장합니다.
    # WHY: 지정된 경로에 데이터를 물리적으로 기록하기 위해서입니다.
    # HOW: np.savez 함수에 경로와 키워드 인자(**numpy_dict)를 전달하여 호출합니다.
    np.savez(path, **numpy_dict)

def load_model(model: Module, path: str):
    """
    WHAT: 저장된 파일로부터 가중치 데이터를 읽어와 모델에 덮어씌우는 함수입니다.
    WHY: 이전에 저장된 모델의 상태를 복구하여 추론(Inference)이나 추가 학습을 이어서 진행하기 위해 필요합니다.
    HOW: np.load로 데이터를 읽어들인 뒤, 포함된 모든 키-값 쌍을 state_dict 형태로 재구성하고, 
         model.load_state_dict를 통해 모델 내부로 데이터를 주입합니다.
    """
    # WHAT: 디스크에서 읽어들인 numpy 데이터를 담는 객체입니다.
    # WHY: 파일에 압축되어 저장된 텐서 값들에 접근하기 위함입니다.
    # HOW: np.load 함수를 사용하여 지정된 경로의 파일을 엽니다.
    data = np.load(path)
    
    # WHAT: 모델에 주입하기 위해 재생성된 상태 딕셔너리입니다.
    # WHY: 모델의 load_state_dict가 요구하는 딕셔너리 형식에 맞추기 위함입니다.
    # HOW: 딕셔너리 컴프리헨션을 사용하여 data 파일 내의 모든 키(files)를 순회하는 루프로 값을 복원합니다.
    state_dict = {k: data[k] for k in data.files}
    
    # WHAT: 모델의 파라미터를 업데이트합니다.
    # WHY: 복원된 상태 딕셔너리 데이터를 실제 모델에 적용하기 위해서입니다.
    # HOW: model.load_state_dict 메서드를 호출하여 상태를 주입합니다.
    model.load_state_dict(state_dict)
