"""
================================================================================
[AMEVA-Forge 역사 기록 (Historical Metadata)]
생성일 (Created): Wed Aug 12 12:14:52 2026 +0900
수정 내역 (Modified):
- Wed Aug 12 12:59:35 2026 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
- Wed Aug 12 12:14:52 2026 +0900: Refactor: Rename AMEVA-Forge and reorganize directories
================================================================================
"""

# WHAT: typing 모듈에서 List 타입을 임포트합니다.
# WHY: 타입 힌팅을 통해 코드의 가독성을 높이고 정적 분석을 용이하게 하기 위함입니다.
# HOW: 반환 값 등의 타입 명시에 사용됩니다.
from typing import List
from collections import OrderedDict
from .tensor import Tensor
from .errors import AMEVAForgeUnsupportedOperationError

# WHAT: 내부 연산 모듈에서 다양한 수학적 연산 함수들을 임포트합니다.
# WHY: 신경망 계층 내에서 순전파 연산을 수행하기 위해 필요한 연산들을 제공하기 때문입니다.
# HOW: 포워드 패스에서 데이터 변환 및 활성화 함수로 호출됩니다.
from .ops import random, zeros, transpose, relu, matmul

# WHAT: 외부 라이브러리인 numpy를 임포트합니다.
# WHY: 수치 해석 및 다차원 배열 연산을 최적화하여 빠르고 효율적으로 처리하기 위함입니다.
# HOW: 초기 가중치 생성, 데이터 타입 변환 등의 기반 연산에 사용됩니다.
import numpy as np

# WHAT: 모든 신경망 모듈의 기본이 되는 베이스 클래스입니다.
# WHY: 계층(layer)들의 파라미터 관리, 상태 저장/불러오기, 훈련/평가 모드 전환 등 공통적인 기능을 제공하기 위해 존재합니다.
# HOW: 사용자가 정의하는 모든 계층이나 모델은 이 클래스를 상속받아 `forward` 메서드를 구현하여 동작합니다.
class Module:
    # WHAT: Module 인스턴스의 초기화 메서드입니다.
    # WHY: 객체가 생성될 때 필요한 내부 상태(서브모듈, 파라미터, 훈련 모드)를 설정하기 위함입니다.
    # HOW: object.__setattr__를 통해 속성 초기화 시 발생할 수 있는 무한 루프나 속성 충돌을 방지합니다.
    def __init__(self):
        # VUL-011 Fix: object.__setattr__로 안전하게 초기화
        # __setattr__ 오버라이드 전에 내부 딕셔너리를 먼저 생성
        
        # WHAT: 현재 모듈에 등록된 하위 모듈들을 저장하는 딕셔너리입니다.
        # WHY: 계층적 구조를 가지는 신경망에서 자식 모듈들을 관리하여 파라미터 추출 및 상태 관리를 용이하게 하기 위함입니다.
        # HOW: 속성 설정 시 값이 Module 인스턴스일 경우 이 딕셔너리에 추가됩니다.
        object.__setattr__(self, '_modules', {})
        
        # WHAT: 현재 모듈에 직접 속한 학습 가능한 파라미터들을 저장하는 딕셔너리입니다.
        # WHY: 그래디언트 업데이트 시 대상이 되는 파라미터들(가중치, 편향 등)을 추적하기 위함입니다.
        # HOW: 속성 설정 시 값이 requires_grad=True인 Tensor일 경우 이 딕셔너리에 추가됩니다.
        object.__setattr__(self, '_params', {})
        
        # WHAT: 모듈의 현재 동작 상태(훈련 중인지 여부)를 나타내는 불리언 변수입니다.
        # WHY: Dropout이나 BatchNorm처럼 훈련 시와 평가 시 동작이 다른 계층들을 제어하기 위함입니다.
        # HOW: True일 경우 훈련 모드, False일 경우 평가(추론) 모드로 동작하게 됩니다.
        object.__setattr__(self, 'training', True)
        
    # WHAT: 모듈과 모든 하위 모듈의 훈련 모드를 설정하는 메서드입니다.
    # WHY: 전체 네트워크의 상태를 일괄적으로 변경하여 훈련 또는 평가에 적합한 동작을 하도록 만들기 위함입니다.
    # HOW: 자기 자신의 상태를 변경한 후, 루프를 돌며 자식 모듈들에 재귀적으로 호출합니다.
    def train(self, mode=True):
        # WHAT: 현재 모듈의 훈련 상태를 인자로 받은 mode로 설정합니다.
        # WHY: 훈련/평가 모드 플래그를 업데이트하기 위함입니다.
        # HOW: self.training 속성에 mode 값을 대입합니다.
        self.training = mode
        
        # WHAT: 자식 모듈들을 순회하며 상태를 전파하는 루프입니다.
        # WHY: 중첩된 신경망 구조에서 하위 모듈들까지 동일한 훈련 상태를 가지도록 하기 위함입니다.
        # HOW: self._modules 딕셔너리의 값들을 하나씩 꺼내어 train(mode)를 호출합니다.
        for m in self._modules.values():
            m.train(mode)
            
    # WHAT: 모듈을 평가(추론) 모드로 전환하는 메서드입니다.
    # WHY: 사용자가 직관적으로 모델을 평가 상태로 바꿀 수 있게 편의성을 제공하기 위함입니다.
    # HOW: 내부적으로 self.train(False)를 호출하여 훈련 상태를 해제합니다.
    def eval(self):
        self.train(False)
    
    # WHAT: 모듈의 순전파 연산을 정의하는 메서드입니다.
    # WHY: 입력 데이터가 이 모듈을 통과할 때 어떤 연산이 일어나는지 명세하기 위함입니다.
    # HOW: 하위 클래스에서 오버라이드하여 구체적인 계산 로직을 구현해야 하며, 기본적으로는 NotImplementedError를 발생시킵니다.
    def forward(self, *args):
        raise NotImplementedError
    
    # WHAT: 모듈 객체를 함수처럼 호출할 수 있게 해주는 매직 메서드입니다.
    # WHY: model(x) 와 같은 직관적인 문법으로 순전파를 실행할 수 있게 지원하기 위함입니다.
    # HOW: 전달받은 인자들을 그대로 forward 메서드로 넘겨주어 반환값을 돌려줍니다.
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    # WHAT: 모델 내부의 모든 학습 가능한 파라미터들을 리스트로 반환하는 메서드입니다.
    # WHY: 옵티마이저(Optimizer)가 어떤 가중치들을 업데이트해야 하는지 알아야 하기 때문입니다.
    # HOW: 자신의 파라미터들을 리스트로 만들고, 모든 자식 모듈을 순회하며 그들의 파라미터도 수집하여 합칩니다.
    def parameters(self) -> List[Tensor]:
        # WHAT: 현재 모듈에 직접 포함된 파라미터들을 리스트 형태로 가져옵니다.
        # WHY: 파라미터 수집의 시작점 역할을 하기 위함입니다.
        # HOW: self._params 딕셔너리의 값들을 list로 변환합니다.
        params = list(self._params.values())
        
        # WHAT: 자식 모듈들을 순회하는 루프입니다.
        # WHY: 하위 모듈들에 숨겨진 파라미터들도 모두 찾아내기 위함입니다.
        # HOW: 각 자식 모듈 m의 parameters() 메서드를 재귀적으로 호출하여 반환된 리스트를 params에 연장(extend)합니다.
        for m in self._modules.values():
            params.extend(m.parameters())
        return params

    def to(self, device: str):
        """
        WHAT: 모델의 모든 파라미터(Parameter), 버퍼(Buffer: running_mean 등), 서브모듈을 지정된 디바이스로 일괄 이동합니다.
        WHY: GPU 가속 또는 CPU 평가를 위해 모델 내 모든 텐서 자원의 연산 장치를 일치시키기 위함입니다.
        HOW: _params와 인스턴스 내 모든 Tensor 객체들을 to(device)로 변환하고 _modules를 재귀 호출합니다.
        """
        if device not in ("cpu", "gpu"):
            from .errors import AMEVAForgeDeviceError
            raise AMEVAForgeDeviceError(
                f"Unsupported device: {device!r}. "
                "Expected 'cpu' or 'gpu'."
            )
        for name, parameter in self._params.items():
            parameter.move_to_(device)
        for name, value in self.__dict__.items():
            if name.startswith('_'):
                continue
            if isinstance(value, Tensor) and name not in self._params:
                value.move_to_(device)
        for module in self._modules.values():
            module.to(device)
        return self

    # WHAT: 모델의 전체 파라미터 상태를 딕셔너리 형태로 추출하는 메서드입니다.
    # WHY: 모델의 가중치를 파일로 저장(Serialization)하거나 다른 모델로 복사할 때 사용하기 위함입니다.
    # HOW: OrderedDict를 생성한 후 자신의 파라미터를 추가하고 자식 모듈을 순회하며 상태를 누적합니다.
    def state_dict(self, prefix='', keep_vars=False):
        # WHAT: 순서가 보장되는 딕셔너리를 임포트하고 생성합니다.
        # WHY: 파라미터의 구조와 순서를 일정하게 유지하기 위함입니다.
        # HOW: collections 모듈에서 OrderedDict를 불러와 인스턴스를 생성합니다.
        from collections import OrderedDict
        state = OrderedDict()
        
        # WHAT: 현재 모듈의 파라미터들을 순회하는 루프입니다.
        # WHY: 상태 사전에 각 파라미터의 이름과 데이터를 저장하기 위함입니다.
        # HOW: self._params.items()를 통해 이름과 파라미터 객체를 가져옵니다.
        for name, param in self._params.items():
            # WHAT: 계층적 구조를 반영한 전체 파라미터 식별 키입니다.
            # WHY: 글로벌 상태 사전 내에서 이름 충돌을 방지하기 위함입니다.
            # HOW: 전달받은 prefix와 현재 파라미터 name을 결합합니다.
            key = prefix + name
            
            if keep_vars:
                # WHAT: Tensor 객체 자체를 보존하여 상태 사전에 저장합니다.
                # WHY: 그래디언트 정보 등 텐서의 고유 메타데이터가 필요할 때 사용하기 위함입니다.
                # HOW: key에 param 객체를 그대로 할당합니다.
                state[key] = param
            else:
                # WHAT: 파라미터의 실제 수치 데이터(NumPy 배열 등)만 추출하여 저장합니다.
                # WHY: 모델 저장 시 불필요한 메타데이터를 제외하고 순수 가중치만 저장하기 위함입니다.
                # HOW: CPU 텐서는 _data 또는 numpy()로 추출하고, GPU 텐서는 _data가 없을 경우 명시적 안내 에러를 발생시킵니다.
                if param.device == 'cpu':
                    state[key] = param._data if param._data is not None else param.numpy()
                elif hasattr(param, '_data') and param._data is not None:
                    state[key] = param._data
                else:
                    from .errors import AMEVAForgeDeviceError
                    raise AMEVAForgeDeviceError(
                        f"state_dict(keep_vars=False) cannot synchronously readback GPU parameter '{key}'. "
                        "Use model.state_dict(keep_vars=True) to retain GPU tensor handles, "
                        "or transfer model to CPU first: model.to('cpu').state_dict()."
                    )
                
        # WHAT: 하위 모듈들을 순회하는 루프입니다.
        # WHY: 트리 구조로 얽힌 모든 파라미터의 상태를 빠짐없이 수집하기 위함입니다.
        # HOW: self._modules를 반복하며 각 모듈에 대해 재귀적으로 state_dict를 호출하고 결과를 업데이트합니다.
        for name, module in self._modules.items():
            if module is not None:
                state.update(module.state_dict(prefix + name + '.', keep_vars))
                
        return state

    # WHAT: 저장된 상태 딕셔너리로부터 모델 파라미터를 복원하는 메서드입니다.
    # WHY: 저장소나 파일에서 불러온 가중치를 현재 모델 객체에 덮어씌워 사용할 수 있게 하기 위함입니다.
    # HOW: 현재 모델의 state_dict를 텐서 형태로 가져온 뒤, 입력된 상태 딕셔너리와 키를 매칭시켜 데이터를 교체합니다.
    def load_state_dict(self, state_dict):
        # WHAT: 현재 모듈의 전체 파라미터 텐서 객체들을 포함하는 딕셔너리를 가져옵니다.
        # WHY: 파라미터 객체의 값을 안전하게 제자리에서(in-place) 덮어씌우기 위해 참조를 확보하기 위함입니다.
        # HOW: keep_vars=True 인자를 주어 데이터뿐 아니라 객체 자체를 리턴받습니다.
        my_state = self.state_dict(keep_vars=True)
        
        # WHAT: 현재 모델 파라미터들을 하나씩 검사하는 루프입니다.
        # WHY: 입력받은 state_dict에 매칭되는 데이터가 있는지 확인하고 복원하기 위함입니다.
        # HOW: my_state 딕셔너리에서 이름과 파라미터 참조를 가져와서 비교합니다.
        for name, param in my_state.items():
            if name in state_dict:
                # WHAT: state_dict에서 복원할 데이터 값을 꺼내옵니다.
                # WHY: 실제 덮어씌울 데이터를 확보하기 위함입니다.
                # HOW: name을 키로 사용하여 값을 조회합니다.
                val = state_dict[name]
                
                if hasattr(val, 'numpy'):
                    # WHAT: 불러온 값이 텐서류 객체일 경우 numpy 배열로 변환합니다.
                    # WHY: 내부 파라미터 데이터는 numpy 배열 기반으로 관리되기 때문입니다.
                    # HOW: numpy() 메서드를 호출하여 다차원 배열을 추출합니다.
                    val = val.numpy()
                    
                # WHAT: 현재 파라미터 객체의 내부 데이터를 새 값으로 대체합니다.
                # WHY: 모델의 가중치를 업데이트하여 복원을 마무리하기 위함입니다.
                # HOW: val을 numpy 배열로 감싸고 기존 데이터 타입과 일치시킨 후 param._data에 덮어씁니다.
                param._data = np.array(val, dtype=param._data.dtype if param._data is not None else np.float32)
    
    # WHAT: 객체의 속성을 설정할 때 호출되는 매직 메서드 오버라이드입니다.
    # WHY: 새로운 속성이 Module인지 Tensor(파라미터)인지 자동으로 감지하여 내부 딕셔너리에 등록하기 위함입니다.
    # HOW: 속성 이름과 값을 분석하여 적절한 내부 컬렉션(_modules 또는 _params)에 추가한 뒤 원본 객체의 __setattr__를 호출합니다.
    def __setattr__(self, name, value):
        if name.startswith('_'):
            # WHAT: 프라이빗(private) 속성에 대한 설정 로직입니다.
            # WHY: 내부 상태 변수들이 _modules나 _params로 오분류되는 것을 방지하기 위함입니다.
            # HOW: 별도 처리 없이 기본 object.__setattr__를 이용해 직접 속성을 설정하고 종료합니다.
            object.__setattr__(self, name, value)
            return
            
        if isinstance(value, Module):
            # WHAT: 할당되는 값이 서브모듈(Module 인스턴스)일 경우의 처리입니다.
            # WHY: 네트워크 구조에 속하는 계층을 모듈 트리에 등록하기 위함입니다.
            # HOW: _modules 딕셔너리에 name을 키로 하여 저장합니다.
            self._modules[name] = value
            
        if isinstance(value, Tensor) and getattr(value, 'requires_grad', False):
            # WHAT: 할당되는 값이 학습을 필요로 하는 텐서(파라미터)일 경우의 처리입니다.
            # WHY: 그래디언트를 계산해야 하는 가중치 및 편향을 파라미터 리스트에 자동으로 등록하기 위함입니다.
            # HOW: _params 딕셔너리에 name을 키로 하여 저장합니다.
            self._params[name] = value
            
        # WHAT: 모든 확인 과정을 거친 후 객체 인스턴스에 실제 속성을 설정합니다.
        # WHY: 클래스 인스턴스가 런타임에 올바른 상태를 유지하도록 하기 위함입니다.
        # HOW: 내장 object.__setattr__ 메서드를 호출합니다.
        object.__setattr__(self, name, value)


# WHAT: 완전 연결 계층(Fully Connected Layer)을 구현한 클래스입니다.
# WHY: 입력 피처와 가중치 간의 선형 변환(Linear Transformation)을 수행하여 특징을 추출하거나 분류를 수행하기 위함입니다.
# HOW: 가중치 행렬 및 편향 벡터를 파라미터로 가지고 입력 데이터와의 행렬 곱을 수행합니다.
class Linear(Module):
    # WHAT: Linear 계층 인스턴스의 초기화 메서드입니다.
    # WHY: 입력 및 출력 차원수를 바탕으로 가중치와 편향 파라미터를 초기화하고 등록하기 위함입니다.
    # HOW: He(Kaiming) 초기화 기법을 사용하여 분산을 보정하고 Tensor 객체로 파라미터를 생성합니다.
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # Kaiming initialization
        # WHAT: 가중치 초기화의 스케일을 설정하는 변수입니다.
        # WHY: 깊은 신경망에서 기울기 소실이나 폭발을 방지하기 위해 분산을 2/in_features로 조정하기 위함입니다.
        # HOW: (2.0 / in_features)의 제곱근을 계산하여 적용합니다.
        scale = (2.0 / in_features) ** 0.5
        
        # WHAT: 정규 분포 기반 무작위 값으로 초기화된 가중치 데이터를 생성하는 변수입니다.
        # WHY: 가중치가 동일한 값으로 시작되는 대칭성을 파괴하고, 학습이 정상적으로 이루어지도록 하기 위함입니다.
        # HOW: numpy.random.randn을 통해 표준정규분포에서 추출한 후 scale을 곱합니다.
        w_data = np.random.randn(out_features, in_features).astype(np.float32) * scale
        
        # WHAT: 모델의 가중치 파라미터 텐서입니다.
        # WHY: 선형 변환 과정에서 입력 데이터와 곱해질 행렬 공간을 유지하기 위함입니다.
        # HOW: requires_grad=True를 주어 그래디언트 계산을 활성화하고 Tensor 인스턴스를 self.weight로 저장합니다.
        self.weight = Tensor(shape=(out_features, in_features), dtype='float32', device='cpu', data=w_data, requires_grad=True)
        
        if bias:
            # WHAT: 모델의 편향(Bias) 파라미터 텐서입니다.
            # WHY: 데이터를 원점으로부터 평행 이동시켜 모델의 표현력을 높이기 위함입니다.
            # HOW: 0으로 초기화된 out_features 크기의 Tensor를 생성하고 requires_grad=True로 설정합니다.
            self.bias = Tensor(shape=(out_features,), dtype='float32', device='cpu',
                             data=np.zeros(out_features, dtype=np.float32), requires_grad=True)
        else:
            self.bias = None
    
    # WHAT: Linear 계층의 순전파 연산을 정의하는 메서드입니다.
    # WHY: 입력 데이터를 받아 선형 결합 수식(Wx + b)을 실제로 계산하기 위함입니다.
    # HOW: 입력 데이터 x와 가중치의 전치행렬 간 행렬 곱(matmul)을 구한 후, 존재한다면 편향을 더합니다.
    def forward(self, x):
        # WHAT: 입력과 가중치의 행렬 곱셈 결과입니다.
        # WHY: 공간 변환 및 특징 추출을 진행하기 위함입니다.
        # HOW: matmul(x, transpose(self.weight)) 연산을 수행합니다.
        out = matmul(x, transpose(self.weight))
        if self.bias is not None:
            # WHAT: 덧셈 연산을 제공하는 함수 임포트입니다.
            # WHY: 편향을 선형 결합 결과에 더해주기 위함입니다.
            # HOW: 브로드캐스팅이 지원되는 내부 ops.add를 호출합니다.
            from .ops import add
            out = add(out, self.bias)  # broadcasting: (batch, out) + (out,)
        return out


# WHAT: ReLU (Rectified Linear Unit) 활성화 함수 계층 클래스입니다.
# WHY: 모델에 비선형성을 부여하여 복잡한 패턴을 학습할 수 있게 하고, 그래디언트 소실 문제를 줄이기 위함입니다.
# HOW: 순전파 시 입력 데이터의 모든 음수를 0으로 변환하고 양수는 그대로 통과시킵니다.
class ReLU(Module):
    # WHAT: ReLU의 순전파 연산입니다.
    # WHY: 입력 텐서 요소별로 비선형 변환을 적용하기 위함입니다.
    # HOW: 내부의 relu 연산 함수에 텐서를 전달하여 결과를 반환합니다.
    def forward(self, x):
        return relu(x)


# WHAT: 시그모이드(Sigmoid) 활성화 함수 계층 클래스입니다.
# WHY: 출력값을 0과 1 사이로 압축하여 확률 등과 같은 스케일로 변환하거나 게이트 제어에 사용하기 위함입니다.
# HOW: 1 / (1 + exp(-x)) 수식을 각 요소에 적용합니다.
class Sigmoid(Module):
    # WHAT: Sigmoid의 순전파 연산입니다.
    # WHY: 텐서의 각 요소에 시그모이드 함수를 통과시키기 위함입니다.
    # HOW: 내부 ops.sigmoid 함수를 임포트하여 적용 결과를 반환합니다.
    def forward(self, x):
        from .ops import sigmoid
        return sigmoid(x)


# WHAT: Tanh (Hyperbolic Tangent) 활성화 함수 계층 클래스입니다.
# WHY: 출력값을 -1과 1 사이로 압축하고, 데이터의 중심을 0으로 맞추어 학습 효율을 개선하기 위함입니다.
# HOW: 하이퍼볼릭 탄젠트 수식을 요소별로 연산합니다.
class Tanh(Module):
    # WHAT: Tanh의 순전파 연산입니다.
    # WHY: 텐서 각 요소를 쌍곡탄젠트 공간으로 맵핑하기 위함입니다.
    # HOW: 내부 ops.tanh_op 함수를 임포트하여 계산된 값을 리턴합니다.
    def forward(self, x):
        from .ops import tanh_op
        return tanh_op(x)


# WHAT: 여러 신경망 계층들을 순차적으로 이어붙여 단일 모듈로 만들어주는 컨테이너 클래스입니다.
# WHY: 복잡한 네트워크 구조를 리스트 형태로 쉽게 정의하고 한 번의 forward 호출로 연속 처리를 가능하게 하기 위함입니다.
# HOW: 초기화 시 인자로 받은 계층들을 내부 딕셔너리에 순서대로 저장하고 순전파 시 차례대로 통과시킵니다.
class Sequential(Module):
    # WHAT: Sequential 인스턴스를 초기화하는 메서드입니다.
    # WHY: 사용자가 제공한 다수의 계층 인스턴스들을 모듈 트리에 등록하기 위함입니다.
    # HOW: 위치 인자(layers)들을 받아 순회하며 문자열로 된 인덱스를 키로 _modules에 저장합니다.
    def __init__(self, *layers):
        super().__init__()
        # WHAT: 전달된 계층들을 순회하며 등록하는 루프입니다.
        # WHY: 순서를 보장하면서 각 레이어 모듈을 자식 모듈로 관리하기 위함입니다.
        # HOW: enumerate를 사용해 인덱스를 얻고, 문자열로 변환하여 키로 사용합니다.
        for i, layer in enumerate(layers):
            self._modules[str(i)] = layer
    
    # WHAT: Sequential 모듈의 순전파 연산입니다.
    # WHY: 등록된 계층들을 순서대로 통과시켜 최종 결과를 얻기 위함입니다.
    # HOW: _modules에 저장된 하위 모듈들을 차례로 호출하며 이전 출력값을 다음 입력값으로 갱신합니다.
    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    def __getitem__(self, idx):
        return list(self._modules.values())[idx]

    def __len__(self):
        return len(self._modules)


class MSELoss(Module):
    """
    Mean Squared Error loss module.
    """
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        from .functional import mse_loss
        return mse_loss(input, target)


# WHAT: 2차원 공간 상의 최대 풀링(Max Pooling) 연산을 수행하는 계층입니다.
# WHY: 공간적 해상도를 줄이면서 중요한 특징(가장 강한 신호)을 보존하여 위치 불변성을 얻고 계산량을 감소시키기 위함입니다.
# HOW: 정해진 커널 크기와 보폭(stride)으로 텐서를 순회하며 최댓값만을 추출합니다.
class MaxPool2d(Module):
    # WHAT: MaxPool2d 클래스의 초기화 메서드입니다.
    # WHY: 풀링 연산의 파라미터(커널 크기, 보폭, 패딩)를 설정하고 저장하기 위함입니다.
    # HOW: 인자로 받은 값을 인스턴스의 속성으로 할당합니다.
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        # WHAT: 풀링 윈도우의 크기입니다.
        # WHY: 추출할 영역의 크기를 결정하기 위함입니다.
        # HOW: 단일 정수 또는 튜플로 저장됩니다.
        self.kernel_size = kernel_size
        
        # WHAT: 윈도우가 이동하는 간격(보폭)입니다.
        # WHY: 출력 특성 맵의 크기와 다운샘플링 비율을 결정하기 위함입니다.
        # HOW: 주어지지 않으면 커널 크기와 동일하게 사용되도록 보존됩니다.
        self.stride = stride
        
        # WHAT: 입력 텐서의 경계에 덧붙일 패딩 크기입니다.
        # WHY: 모서리 부분의 정보 손실을 막거나 출력 크기를 정교하게 맞추기 위함입니다.
        # HOW: 저장해두었다가 순전파 시 연산 함수에 전달됩니다.
        self.padding = padding
        
    # WHAT: MaxPool2d의 순전파 메서드입니다.
    # WHY: 입력 텐서에 2D 맥스 풀링을 적용하기 위함입니다.
    # HOW: ops 모듈의 max_pool2d 함수를 호출하여 계산된 결과를 반환합니다.
    def forward(self, x):
        from .ops import max_pool2d
        return max_pool2d(x, self.kernel_size, self.stride, self.padding)

# WHAT: 2차원 공간 상의 평균 풀링(Average Pooling) 연산을 수행하는 계층입니다.
# WHY: 윈도우 내의 평균값을 취해 특징 맵을 부드럽게 줄이고 전체적인 정보를 유지하기 위함입니다.
# HOW: 커널 크기와 보폭을 지정하고 해당 영역 값들의 평균을 계산합니다.
class AvgPool2d(Module):
    # WHAT: AvgPool2d의 초기화 메서드입니다.
    # WHY: 평균 풀링에 필요한 하이퍼파라미터를 세팅하기 위함입니다.
    # HOW: 커널 크기, 보폭, 패딩을 객체 속성으로 저장합니다.
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
    # WHAT: AvgPool2d의 순전파 메서드입니다.
    # WHY: 입력 피처맵 데이터에 대해 평균 풀링 연산을 수행하기 위함입니다.
    # HOW: 내부의 avg_pool2d 함수에 텐서와 인자들을 전달해 결과를 얻습니다.
    def forward(self, x):
        from .ops import avg_pool2d
        return avg_pool2d(x, self.kernel_size, self.stride, self.padding)

# WHAT: 다차원 텐서를 연속된 1차원 데이터로 펼치는(Flatten) 계층 클래스입니다.
# WHY: 합성곱(CNN) 층을 거친 다차원 피처맵을 완전 연결(Linear) 계층의 입력으로 주입할 수 있도록 형태를 변환하기 위함입니다.
# HOW: start_dim부터 end_dim까지의 차원을 결합하여 새로운 형태의 텐서를 만듭니다.
class Flatten(Module):
    # WHAT: Flatten 계층을 초기화하는 메서드입니다.
    # WHY: 텐서에서 어느 차원 구간을 평탄화할지 범위를 설정하기 위함입니다.
    # HOW: 기본적으로 배치 차원(0)은 유지하고 1번째 차원부터 마지막 차원까지를 속성으로 저장합니다.
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
        
    # WHAT: Flatten 모듈의 순전파 메서드입니다.
    # WHY: 입력 텐서의 형태를 실제로 변환하기 위함입니다.
    # HOW: ops.flatten 함수를 임포트하고 저장된 차원 인자와 함께 호출합니다.
    def forward(self, x):
        from .ops import flatten
        return flatten(x, self.start_dim, self.end_dim)


# WHAT: 2차원 배치 정규화(Batch Normalization 2D) 계층 클래스입니다.
# WHY: 신경망 학습 시 내부 공변량 변화(Internal Covariate Shift)를 줄여 학습 속도와 안정성을 극대화하기 위함입니다.
# HOW: 배치(Batch) 단위로 채널별 평균과 분산을 구해 정규화하고, 학습 가능한 스케일(weight)과 시프트(bias) 파라미터를 적용합니다.
class BatchNorm2d(Module):
    # WHAT: BatchNorm2d 초기화 메서드입니다.
    # WHY: 정규화 시 채널 개수에 맞는 학습 파라미터(감마, 베타)와 이동 평균 데이터를 준비하기 위함입니다.
    # HOW: 가중치(1)와 편향(0)은 학습 파라미터로, 이동 평균과 분산은 비학습 텐서로 초기화하여 속성에 할당합니다.
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        # WHAT: 입력 텐서의 채널 수입니다.
        # WHY: 각 파라미터들의 크기 형상을 결정하기 위함입니다.
        # HOW: 클래스 내부에 저장합니다.
        self.num_features = num_features
        
        # WHAT: 분모에 더해지는 매우 작은 상숫값입니다.
        # WHY: 분산이 0에 가까울 때 발생할 수 있는 0으로 나누기 오류나 수치적 불안정을 방지하기 위함입니다.
        # HOW: 엡실론 값을 멤버 변수로 저장하여 식에 사용합니다.
        self.eps = eps
        
        # WHAT: 이동 평균과 분산을 업데이트할 때 사용되는 모멘텀 수치입니다.
        # WHY: 과거 통계량과 현재 배치의 통계량을 어느 비율로 섞을지 정하여 학습을 안정화하기 위함입니다.
        # HOW: 지수 이동 평균 공식에 모멘텀 가중치로 활용됩니다.
        self.momentum = momentum
        
        # WHAT: 정규화된 값에 곱해지는 스케일(감마) 파라미터 텐서입니다.
        # WHY: 정규화 후에도 네트워크가 기존 데이터의 표현력을 회복할 수 있게 학습시키기 위함입니다.
        # HOW: 채널 크기만큼 1로 초기화되며 그래디언트 계산을 활성화(requires_grad=True)합니다.
        self.weight = Tensor(shape=(num_features,), dtype='float32', device='cpu', data=np.ones(num_features, dtype=np.float32), requires_grad=True)
        
        # WHAT: 정규화된 값에 더해지는 이동(베타) 파라미터 텐서입니다.
        # WHY: 원점을 유연하게 조정하여 모델의 비선형적 성능을 유지하기 위함입니다.
        # HOW: 채널 크기만큼 0으로 초기화되며 학습을 활성화합니다.
        self.bias = Tensor(shape=(num_features,), dtype='float32', device='cpu', data=np.zeros(num_features, dtype=np.float32), requires_grad=True)
        
        # WHAT: 전체 훈련 데이터의 채널별 평균을 추적하는 이동 평균 텐서입니다.
        # WHY: 평가(eval) 시 현재 배치가 아닌 전체 데이터 분포를 기반으로 정규화하기 위함입니다.
        # HOW: 0으로 초기화하고 백프로퍼게이션에 참여하지 않도록 requires_grad=False로 설정합니다.
        self.running_mean = Tensor(shape=(num_features,), dtype='float32', device='cpu', data=np.zeros(num_features, dtype=np.float32), requires_grad=False)
        
        # WHAT: 전체 훈련 데이터의 채널별 분산을 추적하는 이동 분산 텐서입니다.
        # WHY: 평가 시 변동성이 큰 단일 배치의 분산 대신 누적된 전역 분산을 사용하기 위함입니다.
        # HOW: 1로 초기화하고 학습(requires_grad=False) 대상에서 제외합니다.
        self.running_var = Tensor(shape=(num_features,), dtype='float32', device='cpu', data=np.ones(num_features, dtype=np.float32), requires_grad=False)

    # WHAT: BatchNorm2d의 순전파 메서드입니다.
    # WHY: 입력 데이터를 정규화하고 아핀(affine) 변환을 수행하기 위함입니다.
    # HOW: functional 모듈의 batch_norm2d를 호출하며, 훈련/평가 모드 플래그(self.training)를 함께 전달합니다.
    def forward(self, x):
        from .functional import batch_norm2d
        return batch_norm2d(x, self.running_mean, self.running_var, self.weight, self.bias, self.training, self.momentum, self.eps)


# WHAT: 드롭아웃(Dropout) 정규화 기법을 적용하는 계층 클래스입니다.
# WHY: 훈련 중 랜덤하게 특정 뉴런의 출력을 0으로 만들어, 네트워크가 특정 특징에 과적합(Overfitting)되는 것을 방지하기 위함입니다.
# HOW: 정해진 확률 p에 따라 요소들을 마스킹하고, 나머지 값들을 1/(1-p)로 스케일링하여 기댓값을 유지합니다.
class Dropout(Module):
    # WHAT: 드롭아웃 인스턴스를 초기화하는 메서드입니다.
    # WHY: 요소를 0으로 만들 확률값 p를 설정하기 위함입니다.
    # HOW: 파라미터 p를 인스턴스 속성으로 저장합니다.
    def __init__(self, p=0.5):
        super().__init__()
        # WHAT: 뉴런 출력이 무작위로 0이 될 확률입니다.
        # WHY: 모델의 정규화 강도를 조절하기 위함입니다.
        # HOW: self.p 변수에 저장하여 forward 시 사용합니다.
        self.p = p

    # WHAT: 드롭아웃 계층의 순전파 연산입니다.
    # WHY: 훈련 시 무작위 마스킹을 수행하고 추론(eval) 시에는 데이터 손실 없이 그대로 통과시키기 위함입니다.
    # HOW: ops.dropout 함수에 입력 텐서와 함께 현재 모델 상태(self.training)를 넘겨줍니다.
    def forward(self, x):
        from .ops import dropout
        return dropout(x, self.p, self.training)

# WHAT: 2차원 합성곱(Convolution 2D) 계층 클래스입니다.
# WHY: 이미지와 같은 2D 데이터 공간에서 국소적인 특징(가장자리, 질감 등)을 추출하기 위함입니다.
# HOW: 학습 가능한 커널(필터)을 입력 데이터 위로 슬라이딩하면서 합성곱 연산을 수행합니다.
class Conv2d(Module):
    # WHAT: Conv2d 계층의 인스턴스를 초기화하는 메서드입니다.
    # WHY: 입력/출력 채널, 커널 크기 등의 구조적 파라미터를 설정하고 가중치 텐서를 할당하기 위함입니다.
    # HOW: 채널과 커널 크기를 기반으로 He/Kaiming 초기화 범위(k)를 계산하여 가중치와 편향을 균등 분포로 생성합니다.
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        # WHAT: 입력 텐서의 채널 수입니다.
        # WHY: 커널의 깊이 차원을 맞추기 위함입니다.
        # HOW: 속성으로 저장합니다.
        self.in_channels = in_channels
        
        # WHAT: 생성될 출력 피처맵의 채널 수(필터의 개수)입니다.
        # WHY: 추출할 특징의 다양성을 결정하기 위함입니다.
        # HOW: 속성으로 저장합니다.
        self.out_channels = out_channels
        
        # WHAT: 합성곱 커널(필터)의 가로세로 크기입니다.
        # WHY: 수용 영역(Receptive Field)을 결정하기 위함입니다.
        # HOW: 속성으로 저장합니다.
        self.kernel_size = kernel_size
        
        # WHAT: 필터가 이동하는 보폭입니다.
        # WHY: 출력 특성맵의 공간적 크기를 조절하기 위함입니다.
        # HOW: 속성으로 저장합니다.
        self.stride = stride
        
        # WHAT: 입력 주변에 채울 0의 크기입니다.
        # WHY: 합성곱 후 공간 차원이 줄어드는 것을 보정하기 위함입니다.
        # HOW: 속성으로 저장합니다.
        self.padding = padding
        
        import math
        # WHAT: 가중치 초기화 상한/하한값입니다.
        # WHY: 입력 노드 수에 반비례하도록 초기화 범위를 조정해 분산을 유지하기 위함입니다.
        # HOW: 1 / sqrt(in_channels * kernel_size * kernel_size) 수식을 사용합니다.
        k = 1 / math.sqrt(in_channels * kernel_size * kernel_size)
        
        from .ops import random, tensor
        import numpy as np
        
        # WHAT: 합성곱 필터의 가중치 데이터입니다.
        # WHY: 입력으로부터 패턴을 학습하고 추출하는 파라미터로 사용하기 위함입니다.
        # HOW: [-k, k] 사이의 균등 분포에서 난수를 추출해 (out_channels, in_channels, kernel_size, kernel_size) 모양으로 텐서를 생성합니다.
        weight_data = np.random.uniform(-k, k, (out_channels, in_channels, kernel_size, kernel_size)).astype(np.float32)
        self.weight = tensor(weight_data, requires_grad=True)
        
        if bias:
            # WHAT: 채널별 편향(Bias) 파라미터입니다.
            # WHY: 결과값을 이동(shift)시켜 활성화 함수의 비선형성 임계값을 조절하기 위함입니다.
            # HOW: 균등 분포 난수로 1차원 텐서를 생성하고 학습 가능하게 만듭니다.
            bias_data = np.random.uniform(-k, k, (out_channels,)).astype(np.float32)
            self.bias = tensor(bias_data, requires_grad=True)
        else:
            self.bias = None

    # WHAT: Conv2d의 순전파 연산 메서드입니다.
    # WHY: 입력 이미지(텐서)에 대해 실제 필터를 적용하여 특징 맵을 반환하기 위함입니다.
    # HOW: 입력과 가중치의 장치(device)가 다르면 동기화한 뒤, ops 모듈의 conv2d 함수를 호출합니다.
    def forward(self, x: 'Tensor') -> 'Tensor':
        from .ops import conv2d
        from .errors import AMEVAForgeDeviceError
        
        if self.weight.device != x.device:
            raise AMEVAForgeDeviceError(
                f"Conv2d weight device '{self.weight.device}' does not match input device '{x.device}'. "
                f"Call model.to('{x.device}') before executing forward or initializing the optimizer."
            )
            
        if self.bias is not None and self.bias.device != x.device:
            raise AMEVAForgeDeviceError(
                f"Conv2d bias device '{self.bias.device}' does not match input device '{x.device}'. "
                f"Call model.to('{x.device}') before executing forward or initializing the optimizer."
            )
            
        # WHAT: 최종 2D 합성곱 연산을 실행합니다.
        # WHY: 입력과 학습된 필터들 간의 크로스 코릴레이션(cross-correlation) 결과를 구하기 위함입니다.
        # HOW: 내부 C/C++ 기반 또는 최적화된 conv2d 함수로 넘깁니다.
        return conv2d(x, self.weight, self.bias, self.stride, self.padding)

# WHAT: 레이어 정규화(Layer Normalization) 계층 클래스입니다.
# WHY: 시퀀스 데이터나 자연어 처리에서 미니배치 차원이 아닌 피처(레이어) 차원에 대해 정규화를 수행해 학습을 돕기 위함입니다.
# HOW: 각 샘플별로 주어진 차원들(normalized_shape)에 걸쳐 평균과 분산을 구하고 표준화합니다.
class LayerNorm(Module):
    # WHAT: LayerNorm 초기화 메서드입니다.
    # WHY: 정규화할 형태와 엡실론, 학습 가능한 변환 스케일(Affine) 파라미터를 세팅하기 위함입니다.
    # HOW: 정규화 형태를 튜플로 저장하고, 필요시 가중치와 편향 텐서를 1과 0으로 각각 생성합니다.
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            # WHAT: 정규화 형태를 튜플로 강제 변환합니다.
            # WHY: 단일 정수 입력도 내부 연산에서 일관된 튜플 형태로 다루기 위함입니다.
            # HOW: 요소를 하나 가진 튜플로 감쌉니다.
            normalized_shape = (normalized_shape,)
            
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            from .ops import tensor
            import numpy as np
            # WHAT: 정규화 후 분포의 크기를 복원하기 위한 스케일링 파라미터입니다.
            # WHY: 데이터의 중요한 분산 정보가 무분별하게 사라지는 것을 방지하기 위함입니다.
            # HOW: normalized_shape 크기만큼 1.0 값을 가지는 텐서로 초기화합니다.
            self.weight = tensor(np.ones(normalized_shape, dtype=np.float32), requires_grad=True)
            
            # WHAT: 정규화 후 위치를 복원하기 위한 시프트 파라미터입니다.
            # WHY: 데이터의 평균 정보 손실을 보완하기 위함입니다.
            # HOW: 0.0 값을 가지는 텐서로 만듭니다.
            self.bias = tensor(np.zeros(normalized_shape, dtype=np.float32), requires_grad=True)
        else:
            self.weight = None
            self.bias = None

    # WHAT: LayerNorm의 순전파 연산입니다.
    # WHY: 텐서 내 각 샘플 레이어별로 정규화 연산을 수행하기 위함입니다.
    # HOW: 파라미터가 장치에 맞게 준비되었는지 확인 후 내부 functional.layer_norm 함수를 부릅니다.
    def forward(self, x):
        from .functional import layer_norm
        from .errors import AMEVAForgeDeviceError
        
        if self.weight is not None and self.weight.device != x.device:
            raise AMEVAForgeDeviceError(
                f"LayerNorm weight device '{self.weight.device}' does not match input device '{x.device}'. "
                f"Call model.to('{x.device}') before executing forward or initializing the optimizer."
            )
            
        if self.bias is not None and self.bias.device != x.device:
            raise AMEVAForgeDeviceError(
                f"LayerNorm bias device '{self.bias.device}' does not match input device '{x.device}'. "
                f"Call model.to('{x.device}') before executing forward or initializing the optimizer."
            )
            
        return layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

# WHAT: 멀티헤드 어텐션(Multihead Attention) 계층 클래스입니다.
# WHY: 트랜스포머(Transformer) 모델에서 여러 관점(헤드)으로 동시에 시퀀스 내 요소들 간의 상관관계(어텐션)를 파악하기 위함입니다.
# HOW: Query, Key, Value를 각각 선형 변환한 후 여러 개의 헤드로 나누고 어텐션을 병렬 계산한 뒤 다시 결합합니다.
class MultiheadAttention(Module):
    # WHAT: MultiheadAttention의 초기화 메서드입니다.
    # WHY: 임베딩 차원, 헤드 개수를 정하고 프로젝션을 위한 선형 레이어(Linear)를 구성하기 위함입니다.
    # HOW: 각 프로젝션(q, k, v, out)용 Linear 인스턴스를 생성해 속성으로 등록합니다.
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # WHAT: 단일 어텐션 헤드가 처리할 차원의 크기입니다.
        # WHY: 전체 차원을 헤드 수로 균등하게 분할하여 병렬 연산하기 위함입니다.
        # HOW: 전체 임베딩 차원을 헤드 개수로 나눈 몫을 저장합니다.
        self.head_dim = embed_dim // num_heads
        
        # WHAT: Query 텐서를 프로젝션하기 위한 선형 계층입니다.
        # WHY: 입력 데이터를 어텐션 메커니즘을 위한 질의(Query) 공간으로 매핑하기 위함입니다.
        # HOW: embed_dim 크기의 입출력을 갖는 Linear 모듈로 초기화됩니다.
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        
        # WHAT: Key 텐서를 프로젝션하기 위한 선형 계층입니다.
        # WHY: 어텐션에서 질의와 비교될 대상(Key) 공간으로 매핑하기 위함입니다.
        # HOW: Linear(embed_dim, embed_dim)으로 구성됩니다.
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
        
        # WHAT: Value 텐서를 프로젝션하기 위한 선형 계층입니다.
        # WHY: 어텐션 가중치가 곱해져 실제 정보(Value)로 쓰일 공간으로 매핑하기 위함입니다.
        # HOW: Linear(embed_dim, embed_dim)으로 구성됩니다.
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
        
        # WHAT: 여러 헤드에서 합쳐진 결과를 최종 차원으로 복원하는 선형 계층입니다.
        # WHY: 병렬 처리된 다중 관점의 정보를 하나로 융합(mix)하기 위함입니다.
        # HOW: 출력 차원인 embed_dim으로 다시 한번 선형 결합합니다.
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        
    # WHAT: 멀티헤드 어텐션의 순전파 메서드입니다.
    # WHY: 입력된 q, k, v에 대해 실제 스케일드 닷 프로덕트 어텐션 연산을 수행하기 위함입니다.
    # HOW: 입력을 리니어 변환하고 차원을 헤드 단위로 쪼갠 뒤 어텐션을 적용하고 합쳐서 최종 리니어 변환합니다.
    def forward(self, query, key, value, attn_mask=None, is_causal=False):
        from .functional import scaled_dot_product_attention
        from .ops import reshape, permute
        
        # WHAT: 입력 텐서들의 형상(Shape) 정보를 추출합니다.
        # WHY: 차원 변환(reshape) 시 필요한 배치 사이즈(B), 시퀀스 길이(L, S), 임베딩 차원(E)을 알기 위함입니다.
        # HOW: query와 key의 shape 튜플을 언패킹합니다.
        B, L, E = query.shape
        _, S, _ = key.shape
        
        # WHAT: 입력 텐서들을 각 프로젝션 계층을 통과시켜 변환합니다.
        # WHY: 어텐션을 계산할 공간(Sub-space)으로 데이터를 매핑하기 위함입니다.
        # HOW: 미리 정의된 q_proj, k_proj, v_proj를 호출합니다.
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # WHAT: 차원을 분할하고 재배열하여 다중 헤드 형태로 만듭니다.
        # WHY: 1개의 거대한 행렬곱을 num_heads개의 독립적인 행렬곱으로 병렬화하기 위함입니다.
        # HOW: 형상을 (Batch, Length, Heads, HeadDim)으로 바꾼 뒤 (Batch, Heads, Length, HeadDim)으로 치환(permute)합니다.
        q = permute(reshape(q, (B, L, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        k = permute(reshape(k, (B, S, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        v = permute(reshape(v, (B, S, self.num_heads, self.head_dim)), (0, 2, 1, 3))
        
        # WHAT: 어텐션 스코어 및 결과값 계산입니다.
        # WHY: 각 질의에 대해 모든 키와의 유사도를 구해 그 가중치만큼 Value를 혼합하기 위함입니다.
        # HOW: functional의 scaled_dot_product_attention 함수를 호출합니다.
        attn_out = scaled_dot_product_attention(q, k, v, attn_mask, self.dropout, is_causal, self.training)
        
        # WHAT: 다중 헤드 결과를 단일 텐서로 다시 병합합니다.
        # WHY: 다음 레이어로 넘기기 위해 원래 임베딩 차원 형태로 되돌리기 위함입니다.
        # HOW: permute로 헤드와 길이 차원을 되돌린 후 reshape로 묶습니다.
        attn_out = reshape(permute(attn_out, (0, 2, 1, 3)), (B, L, E))
        
        # WHAT: 결합된 결과에 최종 선형 변환을 적용합니다.
        # WHY: 독립적으로 추출된 특징들을 서로 교차(mix)시키고 모델 표현력을 강화하기 위함입니다.
        # HOW: out_proj를 호출하여 결과를 반환합니다.
        return self.out_proj(attn_out)

# WHAT: 트랜스포머 인코더의 단일 레이어 블록 클래스입니다.
# WHY: 자기 주의 메커니즘(Self-Attention)과 피드포워드 네트워크(FFN)를 결합해 시퀀스 내 문맥적 특징을 추출하기 위함입니다.
# HOW: MultiheadAttention, LayerNorm, Linear 레이어들을 순차적이고 잔차 연결(Residual Connection) 형태로 구성합니다.
class TransformerEncoderLayer(Module):
    # WHAT: 인코더 레이어의 초기화 메서드입니다.
    # WHY: 어텐션 계층과 피드포워드 다층 퍼셉트론(MLP) 및 정규화 계층을 생성하기 위함입니다.
    # HOW: 내부 멤버로 각 컴포넌트들을 선언하고 초기화합니다.
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        # WHAT: 멀티헤드 셀프 어텐션 모듈입니다.
        # WHY: 시퀀스 데이터 자신 내부 요소들 간의 관계를 파악하기 위함입니다.
        # HOW: MultiheadAttention 클래스를 생성합니다.
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # WHAT: 피드포워드 신경망의 첫 번째 선형 계층입니다.
        # WHY: 어텐션으로 모인 정보를 고차원(보통 4배) 공간으로 확장해 비선형 패턴을 추출하기 위함입니다.
        # HOW: Linear(d_model, dim_feedforward)로 선언합니다.
        self.linear1 = Linear(d_model, dim_feedforward)
        
        # WHAT: 피드포워드 신경망 내부의 드롭아웃 레이어입니다.
        # WHY: 훈련 중 과적합을 방지하기 위함입니다.
        # HOW: Dropout 모듈을 생성합니다.
        self.dropout = Dropout(dropout)
        
        # WHAT: 피드포워드 신경망의 두 번째 선형 계층입니다.
        # WHY: 확장된 차원을 다시 원래의 임베딩 차원(d_model)으로 축소하기 위함입니다.
        # HOW: Linear(dim_feedforward, d_model)로 선언합니다.
        self.linear2 = Linear(dim_feedforward, d_model)
        
        # WHAT: 어텐션 연산 전/후에 적용할 레이어 정규화 모듈들입니다.
        # WHY: 층이 깊어짐에 따라 데이터 분포가 망가지는 것을 막아 안정적 학습을 보장하기 위함입니다.
        # HOW: LayerNorm 클래스로 두 개의 인스턴스를 생성합니다.
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # WHAT: 각 서브 레이어 결과를 기존 값과 더하기 전 적용하는 드롭아웃입니다.
        # WHY: 잔차 연결 부근에서의 정규화 및 과적합 제어를 위함입니다.
        # HOW: Dropout 객체를 생성합니다.
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
        # WHAT: 피드포워드 신경망의 비선형 활성화 함수입니다.
        # WHY: 단순한 선형 변환이 아닌 복잡한 맵핑 함수를 학습하기 위함입니다.
        # HOW: ReLU 인스턴스를 생성합니다.
        self.activation = ReLU()
        
    # WHAT: 트랜스포머 인코더 레이어의 순전파 연산입니다.
    # WHY: 입력 시퀀스가 어텐션과 피드포워드를 거치며 어떻게 특징이 갱신되는지 정의하기 위함입니다.
    # HOW: Pre-LN 구조와 달리 Post-LN 형태를 차용하여 잔차 연결과 정규화를 적용합니다.
    def forward(self, src, src_mask=None, is_causal=False):
        from .ops import add
        
        # WHAT: 셀프 어텐션 블록의 연산 및 잔차 연결(Residual Connection)입니다.
        # WHY: 현재 입력(src)에 자기 자신과의 문맥 정보(src2)를 결합하기 위함입니다.
        # HOW: q, k, v 모두 src로 넣어 어텐션을 구한 후 dropout을 거쳐 기존 src에 더합니다.
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, is_causal=is_causal)
        src = add(src, self.dropout1(src2))
        
        # WHAT: 첫 번째 서브 레이어 이후의 정규화입니다.
        # WHY: 데이터 스케일을 안정화하기 위함입니다.
        # HOW: norm1을 통과시킵니다.
        src = self.norm1(src)
        
        # WHAT: 피드포워드 네트워크(FFN) 블록의 연산 및 잔차 연결입니다.
        # WHY: 각 토큰 위치마다 개별적으로 비선형성을 가해 고수준 특징을 얻기 위함입니다.
        # HOW: linear1 -> relu -> dropout -> linear2 -> dropout2를 통과시킨 후 이전 src에 더합니다.
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = add(src, self.dropout2(src2))
        
        # WHAT: 두 번째 서브 레이어 이후의 정규화입니다.
        # WHY: 출력값을 한 번 더 안정화하여 다음 레이어로 무사히 전달하기 위함입니다.
        # HOW: norm2를 통과시킨 후 최종 반환합니다.
        src = self.norm2(src)
        
        return src

# WHAT: 포지셔널 인코딩(Positional Encoding) 계층 클래스입니다.
# WHY: 트랜스포머는 RNN처럼 순차적으로 처리하지 않아 순서 정보가 없으므로, 데이터의 위치(순서) 정보를 인공적으로 부여하기 위함입니다.
# HOW: 사인(Sin)과 코사인(Cos) 함수의 서로 다른 주파수를 활용하여 정적 행렬을 만들어 입력에 더합니다.
class PositionalEncoding(Module):
    # WHAT: PositionalEncoding의 초기화 메서드입니다.
    # WHY: 모델 차원과 최대 길이에 맞춰 사인 곡선 기반의 위치 임베딩 매트릭스를 미리 계산해두기 위함입니다.
    # HOW: 수식에 따라 pe 행렬을 계산한 후 학습되지 않는(requires_grad=False) 상수로 저장합니다.
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        import numpy as np
        from .ops import tensor
        
        # WHAT: 위치 정보를 담을 0으로 초기화된 넘파이 행렬입니다.
        # WHY: 미리 최대 길이(max_len)만큼 생성하여 런타임 계산 비용을 아끼기 위함입니다.
        # HOW: shape가 (1, max_len, d_model)인 배열을 생성합니다.
        pe = np.zeros((1, max_len, d_model), dtype=np.float32)
        
        # WHAT: 시퀀스 내의 절대 위치(인덱스) 벡터입니다.
        # WHY: 주기 함수에 입력으로 들어갈 위치 값을 나타내기 위함입니다.
        # HOW: arange로 생성 후 2차원 컬럼 벡터로 변환합니다.
        position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
        
        # WHAT: 차원 위치마다 다르게 적용될 주파수 조절항(Denominator)입니다.
        # WHY: 차원 단위로 주기를 늘려 각기 다른 스케일의 위치 특징을 담기 위함입니다.
        # HOW: 지수 함수를 이용해 10000 기반의 감쇠 계수를 만듭니다.
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * (-np.log(10000.0) / d_model))
        
        # WHAT: 사인과 코사인 함수를 교차하여 매트릭스에 할당합니다.
        # WHY: 짝수 인덱스와 홀수 인덱스 차원에 서로 90도 위상 차를 두어 상대적 거리를 쉽게 학습할 수 있게 하기 위함입니다.
        # HOW: 짝수 인덱스에는 sin, 홀수 인덱스에는 cos 값을 대입합니다.
        pe[0, :, 0::2] = np.sin(position * div_term)
        # WHAT: 계산된 행렬을 텐서화하여 보관하고, 원본 CPU NumPy 데이터를 별도 보존합니다.
        # WHY: 텐서가 GPU로 이동된 후에도 다양한 가변 시퀀스 길이(seq_len) 슬라이스를 안전하게 생성하기 위함입니다.
        # HOW: _pe_raw에 float32 배열을 보관하고 _pe_cache로 재사용합니다.
        self._pe_raw = pe.copy().astype(np.float32)
        self.pe = tensor(pe, requires_grad=False)
        self._pe_cache = OrderedDict()
        
    # WHAT: 포지셔널 인코딩의 순전파 메서드입니다.
    # WHY: 실제 모델 입력 텐서에 순서 정보를 합성하기 위함입니다.
    # HOW: 디바이스/길이별 캐시에서 pe_slice를 조회하고 원래 입력값 x와 더해서(add) 반환합니다.
    def forward(self, x):
        from .ops import add, tensor
        from .errors import AMEVAForgeDeviceError
        if self.pe.device != x.device:
            raise AMEVAForgeDeviceError(
                f"PositionalEncoding buffer device '{self.pe.device}' does not match input device '{x.device}'. "
                f"Call model.to('{x.device}') before executing forward."
            )
            
        seq_len = x.shape[1]
        cache_key = (x.device, seq_len)
        if cache_key in self._pe_cache:
            self._pe_cache.move_to_end(cache_key)
        else:
            if len(self._pe_cache) >= 32:
                old_key, old_tensor = self._pe_cache.popitem(last=False)
                if getattr(old_tensor, 'device', None) == 'gpu':
                    try:
                        old_tensor.dispose()
                    except Exception:
                        pass
            pe_slice_data = self._pe_raw[:, :seq_len, :].astype(np.float32)
            self._pe_cache[cache_key] = tensor(pe_slice_data, device=x.device, requires_grad=False)
            
        pe_slice = self._pe_cache[cache_key]
        return add(x, pe_slice)


# WHAT: 단어 인덱스를 밀집 벡터(Dense Vector)로 변환하는 임베딩(Embedding) 계층입니다.
# WHY: 자연어 처리 등에서 불연속적인 토큰(예: 단어 ID)을 연속적인 고차원 공간으로 매핑하여 신경망이 의미를 학습할 수 있게 하기 위함입니다.
# HOW: (어휘 사전 크기) x (임베딩 차원) 크기의 가중치 행렬을 만들고, 인덱스를 받아 해당하는 벡터를 룩업(Lookup)합니다.
class Embedding(Module):
    # WHAT: 임베딩 계층 초기화 메서드입니다.
    # WHY: 어휘 사전 크기와 임베딩 차원을 받아 가중치 파라미터를 생성하기 위함입니다.
    # HOW: 랜덤 정규분포를 사용해 초기 가중치 행렬을 구성하고 학습 가능한 텐서로 만듭니다.
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        # WHAT: 어휘 사전(Vocabulary)의 총 단어 개수입니다.
        # WHY: 룩업 테이블 행렬의 행(Row) 개수를 결정하기 위함입니다.
        # HOW: 멤버 변수로 저장합니다.
        self.num_embeddings = num_embeddings
        
        # WHAT: 각 단어가 표현될 밀집 벡터의 차원 수입니다.
        # WHY: 룩업 테이블 행렬의 열(Column) 개수를 결정하기 위함입니다.
        # HOW: 멤버 변수로 저장합니다.
        self.embedding_dim = embedding_dim
        
        # Standard normal initialization
        # WHAT: 임베딩 가중치의 초기 데이터 행렬입니다.
        # WHY: 모델이 처음부터 다양한 의미 공간을 탐색하도록 무작위로 분산시키기 위함입니다.
        # HOW: np.random.randn을 통해 (num_embeddings, embedding_dim) 크기의 배열을 생성합니다.
        data = np.random.randn(num_embeddings, embedding_dim).astype(np.float32)
        
        from .ops import tensor
        # WHAT: 임베딩 룩업 테이블 역할을 하는 학습 가능한 텐서입니다.
        # WHY: 훈련 과정을 통해 단어 간의 유사도와 관계를 가중치로 최적화하기 위함입니다.
        # HOW: requires_grad=True로 설정하여 저장합니다.
        self.weight = tensor(data, requires_grad=True)

    # WHAT: 임베딩 계층의 순전파 연산입니다.
    # WHY: 정수 인덱스 시퀀스를 받아 그에 대응하는 실수 벡터 시퀀스로 변환하기 위함입니다.
    # HOW: 내부 ops.embedding 함수를 호출하여 룩업을 수행합니다.
    def forward(self, x):
        from .ops import embedding
        return embedding(self.weight, x)

# WHAT: 기본 순환 신경망 셀(RNN Cell)을 구현한 클래스입니다.
# WHY: 단일 타임스텝(time step)에 대해 입력과 이전 은닉 상태(Hidden State)를 받아 새로운 은닉 상태를 계산하기 위함입니다.
# HOW: 현재 입력값(x)과 이전 상태(hx)를 각각의 가중치로 선형 변환한 후 합치고, Tanh 활성화 함수를 통과시킵니다.
class RNNCell(Module):
    # WHAT: RNNCell 초기화 메서드입니다.
    # WHY: 입력 차원과 은닉 차원에 맞는 가중치(Weight)와 편향(Bias) 파라미터들을 준비하기 위함입니다.
    # HOW: 균등 분포로 입력-은닉 간 가중치, 은닉-은닉 간 가중치, 그리고 각각의 편향을 초기화합니다.
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # WHAT: 파라미터 초기화를 위한 바운딩(k) 값입니다.
        # WHY: 가중치들이 너무 커지거나 작아지지 않도록 은닉층 크기에 반비례하게 제한하기 위함입니다.
        # HOW: 1.0 / hidden_size의 제곱근으로 계산합니다.
        k = (1.0 / hidden_size) ** 0.5
        
        # WHAT: 넘파이 난수로 생성한 가중치 및 편향 데이터 배열들입니다.
        # WHY: 입력-은닉 매핑(w_ih, b_ih)과 은닉-은닉 매핑(w_hh, b_hh)을 무작위로 분산시켜 놓기 위함입니다.
        # HOW: np.random.uniform을 이용해 [-k, k] 범위 내에서 추출합니다.
        w_ih = np.random.uniform(-k, k, (hidden_size, input_size)).astype(np.float32)
        w_hh = np.random.uniform(-k, k, (hidden_size, hidden_size)).astype(np.float32)
        b_ih = np.random.uniform(-k, k, (hidden_size,)).astype(np.float32)
        b_hh = np.random.uniform(-k, k, (hidden_size,)).astype(np.float32)
        
        from .ops import tensor
        # WHAT: 실제 학습에 사용될 텐서 객체들입니다.
        # WHY: 순전파 시 수식을 계산하고, 역전파 시 기울기를 구하기 위함입니다.
        # HOW: 각각을 tensor로 변환하고 requires_grad=True를 켭니다.
        self.weight_ih = tensor(w_ih, requires_grad=True)
        self.weight_hh = tensor(w_hh, requires_grad=True)
        self.bias_ih = tensor(b_ih, requires_grad=True)
        self.bias_hh = tensor(b_hh, requires_grad=True)

    # WHAT: RNNCell의 순전파 메서드입니다.
    # WHY: 타임스텝 t에서의 다음 은닉 상태(h_next)를 구하기 위함입니다.
    # HOW: 수식 h' = tanh(W_ih * x + b_ih + W_hh * h + b_hh)를 계산하여 반환합니다.
    def forward(self, x, hx=None):
        from .ops import zeros, matmul, transpose, add, tanh_op
        
        if hx is None:
            # WHAT: 이전 은닉 상태가 주어지지 않았을 때의 기본값 처리입니다.
            # WHY: 시퀀스의 첫 타임스텝에서는 이전 상태가 없으므로 0으로 초기화하기 위함입니다.
            # HOW: 입력 배치 크기(x.shape[0])와 hidden_size 모양을 갖는 0 텐서를 생성합니다.
            hx = zeros((x.shape[0], self.hidden_size), device=x.device)
            
        # h_next = tanh(x @ weight_ih.T + bias_ih + hx @ weight_hh.T + bias_hh)
        # WHAT: 입력값에 대한 선형 변환 결과(term1)입니다.
        # WHY: 외부 자극(입력)이 현재 상태에 미치는 영향을 계산하기 위함입니다.
        # HOW: 행렬 곱(matmul) 후 편향(bias)을 더합니다.
        term1 = add(matmul(x, transpose(self.weight_ih)), self.bias_ih)
        
        # WHAT: 이전 상태에 대한 선형 변환 결과(term2)입니다.
        # WHY: 과거의 문맥이 현재 상태에 미치는 영향을 계산하기 위함입니다.
        # HOW: 행렬 곱 연산 후 편향을 더합니다.
        term2 = add(matmul(hx, transpose(self.weight_hh)), self.bias_hh)
        
        # WHAT: 최종적으로 새로운 은닉 상태를 생성하는 활성화 함수 통과입니다.
        # WHY: 값의 범위를 [-1, 1]로 제한하여 발산을 막고 비선형성을 더하기 위함입니다.
        # HOW: 두 항을 더한 뒤 tanh_op를 씌웁니다.
        h_next = tanh_op(add(term1, term2))
        return h_next

# WHAT: 장단기 메모리(LSTM) 셀 클래스입니다.
# WHY: 일반적인 RNN의 장기 의존성(Long-Term Dependency) 문제인 기울기 소실을 해결하기 위해 셀 상태(Cell State)와 여러 게이트를 활용하기 위함입니다.
# HOW: 입력, 망각, 출력, 셀 게이트를 동시에 계산하고, 이를 결합하여 새로운 셀 상태와 은닉 상태를 생성합니다.
class LSTMCell(Module):
    # WHAT: LSTMCell 인스턴스를 초기화하는 메서드입니다.
    # WHY: 4개의 게이트(i, f, g, o)를 위한 가중치와 편향 파라미터를 하나로 묶어 할당하기 위함입니다.
    # HOW: 연산 속도 향상을 위해 4배 크기의 파라미터를 한 번에 생성합니다.
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # WHAT: 초기화 상한/하한값 제한 변수입니다.
        # WHY: 가중치 스케일을 통제하기 위함입니다.
        # HOW: 역제곱근 수식을 사용합니다.
        k = (1.0 / hidden_size) ** 0.5
        
        # WHAT: 게이트 4개(입력, 망각, 셀, 출력)의 선형 변환 행렬을 하나로 통합한 배열입니다.
        # WHY: 4번 따로 행렬 곱셈을 하는 것보다 한 번에 크게 곱하는 것이 하드웨어 병렬화에 유리하기 때문입니다.
        # HOW: 4 * hidden_size 크기의 배열로 생성합니다.
        w_ih = np.random.uniform(-k, k, (4 * hidden_size, input_size)).astype(np.float32)
        w_hh = np.random.uniform(-k, k, (4 * hidden_size, hidden_size)).astype(np.float32)
        b_ih = np.random.uniform(-k, k, (4 * hidden_size,)).astype(np.float32)
        b_hh = np.random.uniform(-k, k, (4 * hidden_size,)).astype(np.float32)
        
        from .ops import tensor
        # WHAT: 통합 가중치 및 편향 파라미터 텐서입니다.
        # WHY: 그래디언트 계산을 활성화하여 학습에 사용하기 위함입니다.
        # HOW: tensor로 변환하여 멤버 변수에 등록합니다.
        self.weight_ih = tensor(w_ih, requires_grad=True)
        self.weight_hh = tensor(w_hh, requires_grad=True)
        self.bias_ih = tensor(b_ih, requires_grad=True)
        self.bias_hh = tensor(b_hh, requires_grad=True)

    # WHAT: LSTMCell의 순전파 메서드입니다.
    # WHY: 현재 입력(x)과 이전 상태들(h, c)을 가지고 새로운 상태들(h_next, c_next)을 구하기 위함입니다.
    # HOW: 통합된 선형 연산을 거친 결과를 4등분(chunk)하여 각 게이트에 분배한 뒤 공식을 적용합니다.
    def forward(self, x, hx=None):
        from .ops import zeros, matmul, transpose, add, sigmoid, tanh_op, mul
        
        if hx is None:
            # WHAT: 이전 상태(h, c)가 누락되었을 때의 기본 처리입니다.
            # WHY: 시퀀스 첫 타임스텝에 0으로 된 초기 상태를 주입하기 위함입니다.
            # HOW: h와 c 각각을 0으로 채워진 텐서로 생성합니다.
            h = zeros((x.shape[0], self.hidden_size), device=x.device)
            c = zeros((x.shape[0], self.hidden_size), device=x.device)
        else:
            h, c = hx
            
        # WHAT: 4개의 게이트 입력값을 한 번의 수식으로 구한 결과 행렬입니다.
        # WHY: 효율적인 계산을 위해 결합된 가중치 행렬을 사용해 모두 동시에 계산하기 위함입니다.
        # HOW: W_ih * x + b_ih와 W_hh * h + b_hh를 각각 더합니다.
        gates = add(
            add(matmul(x, transpose(self.weight_ih)), self.bias_ih),
            add(matmul(h, transpose(self.weight_hh)), self.bias_hh)
        )
        
        # WHAT: 입력, 망각, 셀, 출력 게이트의 분할 및 활성화입니다.
        # WHY: 각각의 게이트가 메모리 갱신 과정에서 각자 맡은 역할을 수행하도록 분리하기 위함입니다.
        # HOW: 슬라이싱을 이용해 4등분한 뒤 시그모이드와 쌍곡탄젠트(tanh)를 씌웁니다.
        i_gate = sigmoid(gates[:, 0:self.hidden_size]) # Input gate
        f_gate = sigmoid(gates[:, self.hidden_size:2*self.hidden_size]) # Forget gate
        g_gate = tanh_op(gates[:, 2*self.hidden_size:3*self.hidden_size]) # Cell gate (후보군)
        o_gate = sigmoid(gates[:, 3*self.hidden_size:4*self.hidden_size]) # Output gate
        
        # WHAT: 새로운 셀 상태(Cell State) 업데이트입니다.
        # WHY: 과거의 정보를 지울 부분(f_gate * c)과 새롭게 기억할 부분(i_gate * g_gate)을 합치기 위함입니다.
        # HOW: 요소별 곱(mul)과 덧셈(add)을 사용해 c_next를 구합니다.
        c_next = add(mul(f_gate, c), mul(i_gate, g_gate))
        
        # WHAT: 새로운 은닉 상태(Hidden State) 업데이트입니다.
        # WHY: 다음 타임스텝이나 상위 레이어로 전달할 필터링된 출력을 생성하기 위함입니다.
        # HOW: c_next에 tanh를 적용한 후 출력 게이트와 요소별로 곱합니다.
        h_next = mul(o_gate, tanh_op(c_next))
        
        return h_next, c_next

# WHAT: 다층 시퀀스 처리를 위한 완전한 RNN(Recurrent Neural Network) 모듈입니다.
# WHY: 사용자가 단일 셀을 반복해서 호출하지 않고도, 전체 시퀀스를 한 번에 입력하여 결과를 얻을 수 있도록 감싸기(Wrapper) 위함입니다.
# HOW: RNNCell 인스턴스를 소유하고, 입력 시퀀스를 순회(loop)하며 매 타임스텝마다 셀을 호출해 결과를 누적합니다.
class RNN(Module):
    # WHAT: RNN 모듈 초기화 메서드입니다.
    # WHY: RNNCell을 내부에 생성하고 배치 차원 설정을 기억하기 위함입니다.
    # HOW: 셀 객체를 초기화하여 self.cell에 저장합니다.
    def __init__(self, input_size, hidden_size, batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # WHAT: 입력 텐서의 차원 순서(배치가 먼저 오는지 여부) 플래그입니다.
        # WHY: True일 경우 (Batch, Seq, Feature)로, False일 경우 (Seq, Batch, Feature)로 다루기 위함입니다.
        # HOW: 불리언 값으로 저장합니다.
        self.batch_first = batch_first
        
        # WHAT: 연산을 수행할 내부 코어 유닛입니다.
        # WHY: 타임스텝 단위 계산을 델리게이트(위임)하기 위함입니다.
        # HOW: RNNCell 클래스를 인스턴스화합니다.
        self.cell = RNNCell(input_size, hidden_size)
        
    # WHAT: 전체 시퀀스에 대한 순환 연산을 실행하는 메서드입니다.
    # WHY: 각 시점 데이터들을 차례로 셀에 밀어넣고 출력과 최종 상태를 얻기 위함입니다.
    # HOW: 루프를 돌며 상태를 갱신하고 결과를 리스트에 모은 뒤 결합하여 반환합니다.
    def forward(self, x, hx=None):
        if x.device == "gpu":
            raise AMEVAForgeUnsupportedOperationError(
                "GPU RNN is not supported in Release 1. "
                "RNN requires GPU slice/time-step kernels that are not part of the Release 1 scope."
            )
        from .ops import cat, unsqueeze, permute
        
        if self.batch_first:
            # WHAT: 배치 퍼스트 입력일 경우 시퀀스 길이 축을 0번째로 위치하도록 변환합니다.
            # WHY: 시퀀스 루프를 시간 단위(t)로 쉽게 돌기 위함입니다.
            # HOW: (1, 0, 2) 순서로 permute 시킵니다.
            x = permute(x, (1, 0, 2))
            
        # WHAT: 전체 입력 시퀀스의 길이(타임스텝 수)입니다.
        # WHY: 몇 번 루프를 돌릴지 결정하기 위함입니다.
        # HOW: x.shape[0]으로 알아냅니다.
        seq_len = x.shape[0]
        
        # WHAT: 매 타임스텝의 출력 결과를 저장할 빈 리스트입니다.
        # WHY: 마지막에 하나로 결합(Concat)하기 위함입니다.
        # HOW: 빈 리스트를 만듭니다.
        outputs = []
        h = hx
        
        # WHAT: 시퀀스 길이에 따라 타임스텝을 순회하는 메인 루프입니다.
        # WHY: 과거 정보를 다음 단계로 차례로 넘겨주어 연속적인 추론을 수행하기 위함입니다.
        # HOW: t 인덱스를 사용해 x[t]를 뽑고 셀에 투입합니다.
        for t in range(seq_len):
            x_t = x[t]
            h = self.cell(x_t, h)
            # WHAT: 구해진 은닉 상태를 0번째 축(시간축)을 살려 리스트에 넣습니다.
            # WHY: 나중에 차원(dim=0) 기준으로 이어붙이기 위해 차원을 확장(unsqueeze)해 줍니다.
            # HOW: unsqueeze(h, 0)을 호출합니다.
            outputs.append(unsqueeze(h, 0))
            
        # WHAT: 리스트에 담긴 개별 출력값들을 단일 텐서로 합칩니다.
        # WHY: 네트워크의 최종 반환 형태로 만들기 위함입니다.
        # HOW: 차원 0을 기준으로 cat 연산을 수행합니다.
        out = cat(outputs, dim=0)
        
        if self.batch_first:
            # WHAT: 원래 batch_first 형태였다면 출력 형태도 되돌려줍니다.
            # WHY: 입력 포맷과 출력 포맷의 일관성을 유지하기 위함입니다.
            # HOW: 다시 (1, 0, 2) 순서로 permute합니다.
            out = permute(out, (1, 0, 2))
            
        return out, h

# WHAT: 다층 시퀀스 처리를 위한 완전한 LSTM 모듈입니다.
# WHY: 긴 시퀀스에서도 장기 의존성(Long-term dependency)을 안정적으로 학습하고 추론하기 위함입니다.
# HOW: 내부적으로 LSTMCell을 생성하여 시간축(타임스텝)을 따라 입력을 순회하며 연산합니다.
class LSTM(Module):
    # WHAT: LSTM 모듈의 초기화 메서드입니다.
    # WHY: LSTM 셀 인스턴스를 내부에 구성하고 설정들을 저장하기 위함입니다.
    # HOW: 입력 크기, 은닉 크기를 받아 LSTMCell을 초기화합니다.
    def __init__(self, input_size, hidden_size, batch_first=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # WHAT: 입력 텐서의 배치 축(Batch dimension) 선행 여부를 나타내는 플래그입니다.
        # WHY: 사용자가 (Batch, Seq, ...)와 (Seq, Batch, ...) 중 편리한 데이터 형태를 쓰게 하기 위함입니다.
        # HOW: 불리언 값으로 보관합니다.
        self.batch_first = batch_first
        
        # WHAT: 실제 복잡한 게이트 연산을 담당하는 유닛입니다.
        # WHY: 전체 모듈은 루프만 제어하고 코어 로직은 위임(Delegation)하기 위함입니다.
        # HOW: LSTMCell 인스턴스를 생성해 저장합니다.
        self.cell = LSTMCell(input_size, hidden_size)
        
    # WHAT: 전체 시퀀스에 대한 LSTM 순전파 연산 메서드입니다.
    # WHY: 여러 타임스텝에 걸쳐 입력 데이터를 차례로 처리하여 문맥 결과를 도출하기 위함입니다.
    # HOW: 시퀀스 차원을 기준으로 루프를 반복하며 셀 상태(c)와 은닉 상태(h)를 계속해서 누적 갱신합니다.
    def forward(self, x, hx=None):
        if x.device == "gpu":
            raise AMEVAForgeUnsupportedOperationError(
                "GPU LSTM is not supported in Release 1. "
                "Use CPU LSTM or wait for Release 2 recurrent kernels."
            )
        from .ops import cat, unsqueeze, permute
        if self.batch_first:
            # WHAT: batch_first가 참일 경우 입력 데이터를 타임스텝 우선 순서로 뒤집습니다.
            # WHY: 인덱스로 루프를 돌기 쉽게 x[t] 형태로 맞추기 위함입니다.
            # HOW: permute(1, 0, 2)를 적용합니다.
            x = permute(x, (1, 0, 2))
            
        # WHAT: 처리해야 할 총 타임스텝 수입니다.
        # WHY: 순회할 범위를 알기 위함입니다.
        # HOW: 차원 0의 길이를 가져옵니다.
        seq_len = x.shape[0]
        
        # WHAT: 타임스텝별 반환 은닉 상태를 쌓을 리스트입니다.
        # WHY: 텐서 병합을 위해 임시로 모아두기 위함입니다.
        # HOW: 빈 리스트를 초기화합니다.
        outputs = []
        
        if hx is None:
            # WHAT: 이전 상태(h, c)가 없을 때의 초기화입니다.
            # WHY: None으로 두어 내부 셀(Cell)에서 자체적으로 0 초기화하게 냅두기 위함입니다.
            # HOW: 두 변수에 모두 None을 할당합니다.
            h, c = None, None
        else:
            # WHAT: 튜플(Tuple)로 들어온 이전 상태를 분리합니다.
            # WHY: 각각 은닉 상태(h)와 셀 상태(c)로 나누어 루프 갱신 변수로 쓰기 위함입니다.
            # HOW: 언패킹(Unpacking)을 수행합니다.
            h, c = hx
            
        # WHAT: 시간축(Time step)에 따른 순환 신경망 본체 루프입니다.
        # WHY: 연속적인 과거 정보가 미래로 흐르도록 처리하기 위함입니다.
        # HOW: t가 0부터 seq_len-1까지 순회하며 h, c를 지속적으로 갱신합니다.
        for t in range(seq_len):
            x_t = x[t]
            h, c = self.cell(x_t, (h, c) if h is not None else None)
            
            # WHAT: 얻어낸 은닉 상태를 차원 확장하여 리스트에 추가합니다.
            # WHY: 나중에 시퀀스 차원으로 묶어주기 위해 더미 축(0번)을 추가하기 위함입니다.
            # HOW: unsqueeze를 적용합니다.
            outputs.append(unsqueeze(h, 0))
            
        # WHAT: 모든 타임스텝 출력값을 텐서 덩어리로 합칩니다.
        # WHY: 출력 시퀀스를 구성하기 위함입니다.
        # HOW: 0번 축을 기준으로 cat 연산을 합니다.
        out = cat(outputs, dim=0)
        
        if self.batch_first:
            # WHAT: 원래 batch_first=True 인덱싱으로 데이터를 복구합니다.
            # WHY: 사용자가 주입한 형태와 동일한 출력을 보장하기 위함입니다.
            # HOW: 다시 permute(1, 0, 2)를 적용합니다.
            out = permute(out, (1, 0, 2))
            
        return out, (h, c)

class RMSNorm(Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    LLaMA, Gemma, Mistral 등 현대 LLM 표준 정규화 모듈입니다.
    """
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            from .ops import ones
            self.weight = ones(self.normalized_shape, requires_grad=True)
        else:
            self.weight = None
            
    def forward(self, x):
        from .functional import rms_norm
        return rms_norm(x, weight=self.weight, eps=self.eps)

class RotaryEmbedding(Module):
    """
    Rotary Position Embedding (RoPE) 모듈입니다.
    """
    def __init__(self, dim, base_freq=10000.0):
        super().__init__()
        self.dim = dim
        self.base_freq = base_freq
        
    def forward(self, x, offset_pos=0):
        from .functional import rope
        return rope(x, base_freq=self.base_freq, offset_pos=offset_pos)

class SwiGLU(Module):
    """
    Swish Gated Linear Unit (SwiGLU) Fused Feed-Forward Network Block.
    """
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.gate_proj = Linear(in_features, hidden_features, bias=False)
        self.up_proj = Linear(in_features, hidden_features, bias=False)
        self.down_proj = Linear(hidden_features, in_features, bias=False)
        
    def forward(self, x):
        from .functional import swiglu
        g = self.gate_proj(x)
        u = self.up_proj(x)
        act = swiglu(g, u)
        return self.down_proj(act)

