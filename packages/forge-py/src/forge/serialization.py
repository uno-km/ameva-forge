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
import os
# 신경망 모델의 기본 클래스 및 에러 클래스 임포트
from .nn import Module
from .errors import AMEVAForgeValidationError, AMEVAForgeSecurityError

def save_model(model: Module, path: str):
    """
    WHAT: 주어진 모델의 가중치를 파일로 저장하는 함수입니다.
    WHY: 학습이 완료된 모델을 나중에 다시 사용하거나 배포하기 위해 파일 시스템에 상태를 영구적으로 보존할 필요가 있기 때문입니다.
    HOW: 모델의 state_dict를 호출하여 파라미터 딕셔너리를 얻고, 이를 numpy 포맷(.npz)으로 변환하여 안전하게 저장합니다.
    """
    if not isinstance(path, str) or not path.strip():
        raise AMEVAForgeValidationError(f"[AMEVA-Forge Error] Model save path must be a non-empty string, got: {path}")
    
    if not path.endswith('.npz'):
        path = path + '.npz'

    state_dict = model.state_dict(keep_vars=False)
    numpy_dict = {}
    
    for k, v in state_dict.items():
        if hasattr(v, 'numpy'):
            numpy_dict[k] = v.numpy()
        else:
            numpy_dict[k] = v
            
    try:
        np.savez(path, **numpy_dict)
    except Exception as e:
        raise AMEVAForgeValidationError(f"[AMEVA-Forge Error] Failed to write model weights to '{path}': {e}") from e

def load_model(model: Module, path: str):
    """
    WHAT: 저장된 파일로부터 가중치 데이터를 읽어와 모델에 덮어씌우는 함수입니다.
    WHY: 이전에 저장된 모델의 상태를 복구하여 추론(Inference)이나 추가 학습을 이어서 진행하기 위해 필요합니다.
    HOW: np.load(allow_pickle=False)로 안전하게 데이터를 읽어들인 뒤, 컨텍스트 매니저로 파일 핸들을 즉시 반환하고,
         model.load_state_dict를 통해 모델 내부로 데이터를 주입합니다.
    """
    if not isinstance(path, str) or not path.strip():
        raise AMEVAForgeValidationError(f"[AMEVA-Forge Error] Model load path must be a non-empty string, got: {path}")
    
    if not path.endswith('.npz') and not os.path.exists(path):
        if os.path.exists(path + '.npz'):
            path = path + '.npz'
        else:
            raise AMEVAForgeValidationError(f"[AMEVA-Forge Fatal Error] Model file not found at '{path}' (or '{path}.npz'). Halting execution.")

    try:
        # WHAT: 보안을 위해 allow_pickle=False를 강제하고 컨텍스트 매니저로 파일 핸들을 결정론적으로 닫습니다.
        # WHY: 악성 피클 코드가 포함된 임의 코드 실행(RCE)을 원천 차단하고 OS 파일 핸들 누수를 방지하기 위함입니다.
        with np.load(path, allow_pickle=False) as data:
            state_dict = {k: np.array(data[k]) for k in data.files}
    except ValueError as ve:
        raise AMEVAForgeSecurityError(
            f"[AMEVA-Forge Security Alert] Insecure Pickle payload detected or disallowed object in model file '{path}'. "
            f"Loading halted to prevent Remote Code Execution (RCE): {ve}"
        ) from ve
    except Exception as e:
        raise AMEVAForgeValidationError(f"[AMEVA-Forge Fatal Error] Corrupted or invalid model weight file '{path}': {e}") from e

    model.load_state_dict(state_dict)

