"""
================================================================================
[AMEVA-Forge 역사 기록 (Historical Metadata)]
생성일 (Created): Wed Aug 12 12:14:52 2026 +0900
수정 내역 (Modified):
- Wed Aug 12 12:59:35 2026 +0900: Feat: Introduce v3.0 features (CNN, Pooling, Dropout, Serialization)
- Wed Aug 12 12:23:09 2026 +0900: Docs: Build Apache-style docs and unify tests
- Wed Aug 12 12:14:52 2026 +0900: Refactor: Rename AMEVA-Tensor to AMEVA-Forge and reorganize directories
================================================================================
"""

# WHAT: typing 모듈에서 List 타입을 임포트합니다.
# WHY: 파라미터 리스트의 타입을 명시하여 정적 분석과 코드 가독성을 높이기 위함입니다.
# HOW: 타입 힌트 어노테이션에 List를 사용합니다.
from typing import List, Optional
import math
import asyncio
import numpy as np
from .tensor import Tensor
from .errors import (
    AMEVAForgeDeviceError,
    AMEVAForgeShapeError,
    AMEVAForgeValidationError,
    AMEVAForgeUnsupportedOperationError,
)
from .ops import _require_cpu_data

# WHAT: 모든 최적화 알고리즘의 베이스 클래스인 Optimizer입니다.
# WHY: 다양한 옵티마이저(SGD, Adam 등)가 공통으로 가질 속성과 메서드 인터페이스를 정의하기 위함입니다.
# HOW: 서브클래스에서 이 클래스를 상속받아 step 메서드를 구체화합니다.
class Optimizer:
    # WHAT: Optimizer 인스턴스를 초기화하는 메서드입니다.
    # WHY: 최적화할 파라미터 목록과 학습률을 객체 내부에 저장하기 위함입니다.
    # HOW: 전달받은 파라미터 리스트를 복사하여 저장하고 학습률을 설정합니다.
    def __init__(self, params: List[Tensor], lr: float = 0.01, strict_training: bool = False):
        # WHAT: 최적화 대상이 되는 텐서 파라미터들의 리스트입니다.
        # WHY: 원본 리스트가 외부에서 변경되는 것을 방지하고 안전하게 관리하기 위함입니다.
        # HOW: list() 함수를 통해 새로운 리스트 객체로 복사하여 저장합니다.
        self.params = list(params)
        
        # WHAT: 학습률(Learning Rate)입니다.
        # WHY: 각 파라미터 업데이트 시 그래디언트를 얼마나 반영할지 스텝 크기를 결정하기 위함입니다.
        # HOW: 속성으로 저장되어 step 연산에 곱해집니다.
        self.lr = lr

        # WHAT: 엄격 학습 모드(Strict Training Mode) 플래그입니다.
        # WHY: NaN/Inf 그래디언트 발생 시 즉시 Fail-Fast 예외를 발생시켜 학습 발산을 조기 차단하기 위함입니다.
        self.strict_training = strict_training
    
    # WHAT: 파라미터 업데이트를 수행하는 메서드 인터페이스입니다.
    # WHY: 각 옵티마이저마다 고유의 업데이트 규칙(규칙)을 적용하기 위함입니다.
    # HOW: 베이스 클래스에서는 NotImplementedError를 발생시키며, 하위 클래스에서 오버라이드해야 합니다.
    def step(self):
        raise NotImplementedError
    
    def _active_devices(self):
        return {
            p.device
            for p in self.params
            if p.grad is not None
        }

    def _validate_param_grad_pair(self, p: Tensor) -> None:
        """파라미터와 gradient의 장치, shape, dtype 계약을 검증한다."""
        if p.grad is None:
            return
        if p.device != p.grad.device:
            raise AMEVAForgeDeviceError(
                f"Parameter/gradient device mismatch: {p.device} != {p.grad.device}"
            )
        if tuple(p.shape) != tuple(p.grad.shape):
            raise AMEVAForgeShapeError(
                f"Parameter/gradient shape mismatch: {p.shape} != {p.grad.shape}"
            )
        if p.dtype != p.grad.dtype:
            raise AMEVAForgeDeviceError(
                f"Parameter/gradient dtype mismatch: {p.dtype} != {p.grad.dtype}"
            )

    # WHAT: 등록된 모든 파라미터의 그래디언트를 초기화(None)하는 메서드입니다.
    # WHY: 새로운 미니배치의 학습을 시작할 때 이전 배치의 누적된 그래디언트를 지우기 위함입니다.
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                if getattr(p.grad, 'device', None) == 'gpu' and getattr(p.grad, '_handle', None) is not None:
                    try:
                        p.grad.dispose()
                    except Exception:
                        pass
                p.grad = None


# WHAT: 확률적 경사 하강법(Stochastic Gradient Descent, SGD) 옵티마이저 클래스입니다.
# WHY: 모멘텀(Momentum)이 적용될 수 있는 기본적인 기울기 하강 업데이트를 수행하기 위함입니다.
# HOW: Optimizer를 상속받아 step 메서드를 구현하고, 속도를 추적하는 velocity 배열을 관리합니다.
class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, strict_training: bool = False):
        super().__init__(params, lr, strict_training=strict_training)
        self.momentum = momentum
        self.velocity = [None] * len(self.params)
    
    def step(self, strict: Optional[bool] = None):
        """
        CPU parameter 전용 동기 SGD step.

        GPU parameter는 readback이 비동기이므로 이 메서드에서 처리하지 않는다.
        GPU 학습에서는 반드시 `await optimizer.step_async()`를 사용한다.
        """
        use_strict = self.strict_training if strict is None else strict
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            self._validate_param_grad_pair(p)

            if p.device == "gpu":
                raise AMEVAForgeDeviceError(
                    "SGD.step() is CPU-only for GPU-backed parameters. "
                    "Use: await optimizer.step_async()"
                )

            grad_data = p.grad.numpy()
            if use_strict and not np.isfinite(grad_data).all():
                raise AMEVAForgeValidationError(
                    "Non-finite gradient (NaN/Inf) detected in strict training mode."
                )

            param_data = p.numpy()

            if self.momentum > 0.0:
                if self.velocity[i] is None:
                    self.velocity[i] = grad_data.copy()
                else:
                    self.velocity[i] = (
                        self.momentum * self.velocity[i] + grad_data
                    )
                update = self.velocity[i]
            else:
                update = grad_data

            p._data = (param_data - self.lr * update).astype(np.float32)
            p._version += 1
            p.grad = None

    async def step_async(self, strict: Optional[bool] = None):
        """
        CPU와 GPU parameter를 모두 처리하는 공식 비동기 SGD step.

        GPU parameter는 기존 AXPY WGSL을 통해 readback 없이 in-place 갱신한다.
        Release 1에서는 GPU momentum을 지원하지 않으며, 단일 스텝 내 혼합 장치를 허용하지 않는다.
        """
        use_strict = self.strict_training if strict is None else strict
        if not math.isfinite(self.lr) or self.lr <= 0.0:
            raise ValueError(f"lr must be finite and > 0, got {self.lr}")

        active_devices = self._active_devices()
        if len(active_devices) > 1:
            raise AMEVAForgeDeviceError(
                "Mixed CPU/GPU parameters in one SGD step are not supported in Release 1. "
                "Use one optimizer per device."
            )
        from .graph import GraphBuilder
        from .bridge import js_execute_graph

        builder = GraphBuilder()
        cpu_updates = []
        param_out_map = []
        param_entries = []

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            self._validate_param_grad_pair(p)

            if p.device == "cpu":
                grad_data = p.grad.numpy()
                if use_strict and not np.isfinite(grad_data).all():
                    raise AMEVAForgeValidationError(
                        "Non-finite gradient (NaN/Inf) detected in strict training mode."
                    )
                param_data = p.numpy()

                if self.momentum > 0.0:
                    if self.velocity[i] is None:
                        self.velocity[i] = grad_data.copy()
                    else:
                        self.velocity[i] = (
                            self.momentum * self.velocity[i] + grad_data
                        )
                    update = self.velocity[i]
                else:
                    update = grad_data

                cpu_updates.append((p, param_data, update))
                continue

            await p.realize()
            await p.grad.realize()

            if p._handle is None or p.grad._handle is None:
                raise AMEVAForgeDeviceError(
                    "GPU SGD requires realized parameter and gradient handles."
                )

            if use_strict:
                grad_check = await p.grad.numpy_async()
                if not np.isfinite(grad_check).all():
                    raise AMEVAForgeValidationError(
                        "Non-finite gradient (NaN/Inf) detected in strict training mode on GPU."
                    )

            num_elements = int(np.prod(p.shape, dtype=np.int64))
            if num_elements <= 0:
                raise AMEVAForgeShapeError(
                    f"GPU SGD does not support empty parameter: shape={p.shape}"
                )

            param_id = builder.add_load(p.shape, p._handle)
            grad_id = builder.add_load(p.grad.shape, p.grad._handle)

            if self.momentum > 0.0:
                if self.velocity[i] is None or not isinstance(self.velocity[i], Tensor) or self.velocity[i].device != "gpu":
                    from .ops import zeros
                    self.velocity[i] = zeros(p.shape, device="gpu", dtype="float32")
                await self.velocity[i].realize()
                vel_id = builder.add_load(self.velocity[i].shape, self.velocity[i]._handle)
                param_entries.append((p, num_elements, grad_id, param_id, vel_id))
            else:
                param_entries.append((p, num_elements, grad_id, param_id, None))

        for entry in param_entries:
            if self.momentum > 0.0:
                p, num_elements, grad_id, param_id, vel_id = entry
                out_id = builder.add_op(
                    "sgd_momentum_step",
                    p.shape,
                    [param_id, grad_id, vel_id],
                    [float(self.lr), float(self.momentum)],
                )
            else:
                p, num_elements, grad_id, param_id, _ = entry
                out_id = builder.add_op(
                    "axpy",
                    p.shape,
                    [grad_id, param_id],
                    [num_elements, float(self.lr)],
                )
            param_out_map.append((p, out_id))

        # CPU 계산은 검증이 끝난 뒤 원자적으로 반영한다.
        for p, param_data, update in cpu_updates:
            p._data = (param_data - self.lr * update).astype(np.float32)
            p._version += 1
            p.grad = None

        # GPU 그래프는 단일 일괄 FFI 호출로 실행하여 브리지 오버헤드를 최소화한다.
        if param_out_map:
            instructions, inputs = builder.compile()
            result = await js_execute_graph(instructions, inputs)

            for p, out_id in param_out_map:
                returned_handle = result.get(str(out_id)) or result.get(out_id)

                # in-place 계약이므로 같은 parameter handle을 반환해야 한다.
                if returned_handle != p._handle:
                    raise AMEVAForgeDeviceError(
                        "Optimizer contract violation: optimizer returned a different handle."
                    )

                p._version += 1
                if p.grad is not None and getattr(p.grad, 'device', None) == 'gpu':
                    try:
                        p.grad.dispose()
                    except Exception:
                        pass
                p.grad = None


# WHAT: Adam(Adaptive Moment Estimation) 옵티마이저 클래스입니다.
# WHY: 1차 모멘트(평균)와 2차 모멘트(분산)를 추정하여 각 파라미터마다 적응형(adaptive) 학습률을 적용하기 위함입니다.
# HOW: 그래디언트의 지수 이동 평균을 두 가지 형태로 누적하고 편향이 보정된(bias-corrected) 값으로 가중치를 업데이트합니다.
class Adam(Optimizer):
    # WHAT: Adam 인스턴스 초기화 메서드입니다.
    # WHY: Adam 알고리즘에 필요한 하이퍼파라미터(베타, 엡실론)를 설정하고 모멘트 저장 공간을 할당하기 위함입니다.
    # HOW: 상위 초기화 후 beta, eps 등을 저장하고 상태 변수 리스트를 생성합니다.
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay: float = 0.0):
        super().__init__(params, lr)
        # WHAT: 1차 및 2차 모멘트 추정을 위한 감쇠율(decay rate)입니다.
        # WHY: 과거의 그래디언트 정보를 어느 정도 비율로 반영할지 결정하기 위함입니다.
        # HOW: 튜플에서 언패킹하여 각각 beta1, beta2로 저장합니다.
        self.beta1, self.beta2 = betas
        
        # WHAT: 분모가 0이 되는 것을 방지하는 작은 상수입니다.
        # WHY: 수치적 안정성을 보장하기 위함입니다.
        # HOW: 업데이트 식의 제곱근 항에 더해집니다.
        self.eps = eps
        self.weight_decay = float(weight_decay)
        
        # WHAT: 파라미터별 1차 모멘트(평균 추정치)를 저장하는 리스트입니다.
        # WHY: 각 방향별 모멘텀을 추적하기 위함입니다.
        # HOW: None으로 초기화된 리스트를 파라미터 개수만큼 생성합니다.
        self.m = [None] * len(self.params)
        
        # WHAT: 파라미터별 2차 모멘트(비중심 분산 추정치)를 저장하는 리스트입니다.
        # WHY: 그래디언트의 크기 변화에 따라 학습률을 조절하기 위함입니다.
        # HOW: None으로 초기화된 리스트를 파라미터 개수만큼 생성합니다.
        self.v = [None] * len(self.params)
        
        # WHAT: 최적화 스텝을 밟은 횟수(타임스텝)입니다.
        # WHY: 편향 보정(bias correction) 시 t승을 계산하기 위함입니다.
        # HOW: 0으로 시작하여 step()마다 1씩 증가합니다.
        self.t = 0
    
    # WHAT: Adam의 파라미터 업데이트를 1스텝 진행하는 메서드입니다.
    # WHY: 각 파라미터마다 적응형 학습률 수식을 계산하여 가중치를 최적화하기 위함입니다.
    # HOW: 타임스텝을 올리고 각 파라미터에 대해 m, v를 갱신 및 보정한 뒤 값을 차감합니다.
    def step(self):
        # WHAT: 타임스텝 증가 연산입니다.
        # WHY: 편향 보정을 정확히 계산하기 위해 현재 몇 번째 업데이트인지 기록하기 위함입니다.
        # HOW: self.t 값에 1을 더합니다.
        self.t += 1
        
        # WHAT: 파라미터들을 순회하며 Adam 수식을 개별적으로 적용하는 루프입니다.
        # WHY: 신경망 전체 가중치를 갱신하기 위함입니다.
        # HOW: enumerate를 통해 인덱스와 파라미터 객체를 순회합니다.
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            if p.device == "gpu":
                raise AMEVAForgeDeviceError(
                    "Adam optimizer does not support synchronous GPU step in Release 1. "
                    "Use SGD.step_async() for GPU models, or run Adam on CPU tensors."
                )
            else:
                g = _require_cpu_data(p.grad, "p.grad")
                param_data = _require_cpu_data(p, "p")
            
            if self.m[i] is None:
                # WHAT: 처음 업데이트 시 모멘트 배열을 0으로 초기화합니다.
                # WHY: 메모리가 아직 할당되지 않은 상태이므로 그래디언트와 같은 크기의 0 행렬을 만들기 위함입니다.
                # HOW: np.zeros_like 함수를 사용합니다.
                self.m[i] = np.zeros_like(g)
                self.v[i] = np.zeros_like(g)
            
            # WHAT: 1차 및 2차 모멘트의 지수 이동 평균을 업데이트합니다.
            # WHY: 현재 그래디언트의 방향과 크기를 과거 통계치에 반영하기 위함입니다.
            # HOW: beta 상수와 결합된 수식으로 m과 v를 갱신합니다.
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g * g
            
            # WHAT: 학습 초기에 모멘트가 0으로 편향되는 것을 방지하는 편향 보정 연산입니다.
            # WHY: 초기에 m과 v가 너무 작게 측정되어 학습이 비정상적으로 튀는 것을 막기 위함입니다.
            # HOW: 각각 1 - beta^t 로 나누어 m_hat, v_hat을 계산합니다.
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # WHAT: 최종적으로 파라미터를 업데이트하는 수식입니다 (Decoupled Weight Decay 지원).
            # WHY: 보정된 모멘트를 사용하여 적응적으로 스텝을 이동하기 위함입니다.
            # HOW: 파라미터 데이터에서 감쇠를 적용한 뒤 `lr * m_hat / (sqrt(v_hat) + eps)`를 차감합니다.
            if self.weight_decay > 0.0:
                param_data = param_data * (1.0 - self.lr * self.weight_decay)
            param_data = param_data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            
            # WHAT: 갱신된 데이터를 텐서에 덮어쓰고 기울기를 비웁니다.
            # WHY: 다음 스텝을 위해 상태를 최신화하기 위함입니다.
            # HOW: astype(np.float32)로 형변환 후 p._data에 할당하고 grad를 초기화합니다.
            p._data = param_data.astype(np.float32)
            p._version += 1
            p.grad = None

    async def step_async(self):
        """
        CPU와 GPU parameter를 모두 지원하는 공식 비동기 Adam step.
        CPU 파라미터는 로컬 NumPy 벡터 연산으로, GPU 파라미터는 1-Pass 융합 adam_step WGSL 커널을 통해 in-place 갱신된다.
        """
        has_gpu = any(p.grad is not None and p.device == "gpu" for p in self.params)
        if not has_gpu:
            self.step()
            return

        self.t += 1

        from .graph import GraphBuilder
        from .bridge import js_execute_graph

        builder = GraphBuilder()
        param_out_map = []
        param_entries = []
        cpu_updates = []

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            self._validate_param_grad_pair(p)

            if p.device == "cpu":
                g = _require_cpu_data(p.grad, "p.grad")
                param_data = _require_cpu_data(p, "p")

                if self.m[i] is None or not isinstance(self.m[i], np.ndarray):
                    self.m[i] = np.zeros_like(g)
                    self.v[i] = np.zeros_like(g)

                if self.weight_decay != 0.0:
                    g = g + self.weight_decay * param_data

                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)

                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)

                update = m_hat / (np.sqrt(v_hat) + self.eps)
                cpu_updates.append((p, param_data, update))
                continue

            await p.realize()
            await p.grad.realize()

            if p._handle is None or p.grad._handle is None:
                raise AMEVAForgeDeviceError(
                    "GPU Adam requires realized parameter and gradient handles."
                )

            from .ops import zeros
            if self.m[i] is None or not isinstance(self.m[i], Tensor) or self.m[i].device != "gpu":
                self.m[i] = zeros(p.shape, device="gpu", dtype="float32")
            if self.v[i] is None or not isinstance(self.v[i], Tensor) or self.v[i].device != "gpu":
                self.v[i] = zeros(p.shape, device="gpu", dtype="float32")

            await self.m[i].realize()
            await self.v[i].realize()

            param_id = builder.add_load(p.shape, p._handle)
            grad_id = builder.add_load(p.grad.shape, p.grad._handle)
            m_id = builder.add_load(self.m[i].shape, self.m[i]._handle)
            v_id = builder.add_load(self.v[i].shape, self.v[i]._handle)

            param_entries.append((p, param_id, grad_id, m_id, v_id))

        beta1_power = float(self.beta1 ** self.t)
        beta2_power = float(self.beta2 ** self.t)

        for p, param_id, grad_id, m_id, v_id in param_entries:
            out_id = builder.add_op(
                "adam_step",
                p.shape,
                [param_id, grad_id, m_id, v_id],
                [
                    float(self.lr),
                    float(self.beta1),
                    float(self.beta2),
                    float(self.eps),
                    beta1_power,
                    beta2_power,
                    float(self.weight_decay),
                ],
            )
            param_out_map.append((p, out_id))

        # CPU 계산은 검증이 끝난 뒤 원자적으로 반영한다.
        for p, param_data, update in cpu_updates:
            p._data = (param_data - self.lr * update).astype(np.float32)
            p._version += 1
            p.grad = None

        if param_out_map:
            instructions, inputs = builder.compile()
            result = await js_execute_graph(instructions, inputs)

            for p, out_id in param_out_map:
                returned_handle = result.get(str(out_id)) or result.get(out_id)
                if returned_handle != p._handle:
                    raise AMEVAForgeDeviceError(
                        "Adam contract violation: optimizer returned a different handle."
                    )
                p._version += 1
                if p.grad is not None and getattr(p.grad, 'device', None) == 'gpu':
                    try:
                        p.grad.dispose()
                    except Exception:
                        pass
                p.grad = None


# WHAT: 파라미터들의 그래디언트 글로벌 L2 노름(Norm)을 제한(Clip)하는 동기 함수입니다.
# WHY: RNN이나 깊은 신경망에서 그래디언트 폭발(Gradient Exploding) 문제를 방지하여 학습을 안정화하기 위함입니다.
# HOW: 모든 CPU 그래디언트의 제곱합을 구해 노름을 계산하고, max_norm을 넘으면 그 비율만큼 전체 기울기를 축소합니다.
def clip_grad_norm(parameters: List[Tensor], max_norm: float) -> float:
    total_norm = 0.0
    for p in parameters:
        if p.device == "gpu" or (p.grad is not None and p.grad.device == "gpu"):
            raise AMEVAForgeDeviceError(
                "clip_grad_norm() is synchronous and supported only for CPU parameters. "
                "For GPU tensors in Pyodide/WebGPU, use 'await clip_grad_norm_async(parameters, max_norm)'."
            )
        if p.grad is not None:
            g = _require_cpu_data(p.grad, "p.grad")
            total_norm += float(np.sum(g ** 2))
                
    total_norm = float(np.sqrt(total_norm))
    clip_coef = max_norm / (total_norm + 1e-6)
    
    if clip_coef < 1.0:
        for p in parameters:
            if p.grad is not None:
                g = _require_cpu_data(p.grad, "p.grad")
                p.grad._data = (g * clip_coef).astype(np.float32)
    return total_norm


# WHAT: WebGPU 및 Pyodide 비동기 환경을 지원하는 GPU/CPU 통합 비동기 Gradient Norm Clipping 함수입니다.
async def clip_grad_norm_async(parameters: List[Tensor], max_norm: float) -> float:
    valid_params = [p for p in parameters if p.grad is not None]
    if not valid_params:
        return 0.0

    async def fetch_grad(grad_tensor: Tensor) -> np.ndarray:
        if grad_tensor.device == "gpu":
            return await grad_tensor.numpy_async()
        return _require_cpu_data(grad_tensor, "p.grad")

    grads_data = await asyncio.gather(*(fetch_grad(p.grad) for p in valid_params))
    
    total_norm = 0.0
    for g in grads_data:
        total_norm += float(np.sum(g ** 2))
                
    total_norm = float(np.sqrt(total_norm))
    clip_coef = max_norm / (total_norm + 1e-6)
    
    if clip_coef < 1.0:
        for p, g in zip(valid_params, grads_data):
            scaled = (g * clip_coef).astype(np.float32)
            if p.grad.device == "gpu":
                old_grad = p.grad
                from .tensor import tensor as create_tensor
                p.grad = create_tensor(scaled, device="gpu")
                if hasattr(old_grad, 'dispose'):
                    old_grad.dispose()
            else:
                p.grad._data = scaled
    return total_norm


# WHAT: 개별 그래디언트 요소의 최댓값/최솟값을 직접 자르는(Value Clipping) 동기 함수입니다.
# WHY: 매우 큰 특정 그래디언트 값이 전체 학습을 망치는 것을 방지하기 위함입니다.
# HOW: 각 그래디언트 요소를 [-clip_value, clip_value] 범위 내로 제한(clip)합니다.
def clip_grad_value(parameters: List[Tensor], clip_value: float) -> None:
    for p in parameters:
        if p.device == "gpu" or (p.grad is not None and p.grad.device == "gpu"):
            raise AMEVAForgeDeviceError(
                "clip_grad_value() is synchronous and supported only for CPU parameters. "
                "For GPU tensors in Pyodide/WebGPU, use 'await clip_grad_value_async(parameters, clip_value)'."
            )
        if p.grad is not None:
            g = _require_cpu_data(p.grad, "p.grad")
            p.grad._data = np.clip(g, -clip_value, clip_value).astype(np.float32)


# WHAT: WebGPU 및 Pyodide 비동기 환경을 지원하는 GPU/CPU 통합 비동기 Gradient Value Clipping 함수입니다.
async def clip_grad_value_async(parameters: List[Tensor], clip_value: float) -> None:
    valid_params = [p for p in parameters if p.grad is not None]
    if not valid_params:
        return

    async def fetch_grad(grad_tensor: Tensor) -> np.ndarray:
        if grad_tensor.device == "gpu":
            return await grad_tensor.numpy_async()
        return _require_cpu_data(grad_tensor, "p.grad")

    grads_data = await asyncio.gather(*(fetch_grad(p.grad) for p in valid_params))
    for p, g in zip(valid_params, grads_data):
        clipped = np.clip(g, -clip_value, clip_value).astype(np.float32)
        if p.grad.device == "gpu":
            old_grad = p.grad
            from .tensor import tensor as create_tensor
            p.grad = create_tensor(clipped, device="gpu")
            if hasattr(old_grad, 'dispose'):
                old_grad.dispose()
        else:
            p.grad._data = clipped

# WHAT: 정해진 에포크 주기마다 학습률을 단계적으로 감소시키는 스케줄러입니다.
# WHY: 학습 후반부에 학습률을 낮춰 더 세밀한 최적화 지점(Global Minimum)에 도달하게 하기 위함입니다.
# HOW: step이 호출될 때마다 카운트를 올리고 주기(step_size)에 도달하면 lr에 gamma를 곱합니다.
class StepLR:
    # WHAT: StepLR 스케줄러의 초기화 메서드입니다.
    # WHY: 제어할 옵티마이저와 감소 주기, 감소 비율을 설정하기 위함입니다.
    # HOW: 속성들을 객체 내부에 저장합니다.
    def __init__(self, optimizer, step_size, gamma=0.1):
        # WHAT: 대상이 되는 옵티마이저 인스턴스입니다.
        # WHY: 옵티마이저 내부에 저장된 lr 값을 직접 수정하기 위함입니다.
        # HOW: 참조를 저장합니다.
        self.optimizer = optimizer
        
        # WHAT: 학습률을 감소시킬 에포크 주기입니다.
        # WHY: 몇 에포크마다 감쇠시킬지 판단하는 기준이 되기 때문입니다.
        # HOW: 변수로 저장됩니다.
        self.step_size = step_size
        
        # WHAT: 학습률을 감소시킬 비율(감쇠율)입니다.
        # WHY: 기존 학습률에 곱해져 값을 줄이는 강도를 결정하기 위함입니다.
        # HOW: 보통 0.1 등의 값을 저장합니다.
        self.gamma = gamma
        
        # WHAT: 지금까지 진행된 에포크 수입니다.
        # WHY: 주기에 도달했는지 확인하는 카운터로 사용하기 위함입니다.
        # HOW: 0으로 시작합니다.
        self.last_epoch = 0
        
    # WHAT: 1에포크가 끝났을 때 스케줄러를 한 스텝 전진시키는 메서드입니다.
    # WHY: 조건을 검사하고 필요 시 학습률을 감소시키기 위함입니다.
    # HOW: last_epoch를 1 늘리고, step_size로 나누어 떨어지면 lr에 gamma를 곱합니다.
    def step(self, metrics=None):
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma

# WHAT: 코사인 곡선을 따라 학습률을 부드럽게 감소시키는 스케줄러입니다.
# WHY: 웜 리스타트(Warm restart) 효과나 부드러운 하강을 통해 로컬 미니멈을 효율적으로 탈출/수렴하기 위함입니다.
# HOW: 반주기(T_max) 동안 base_lr에서 eta_min까지 코사인 함수 모양으로 학습률을 조절합니다.
class CosineAnnealingLR:
    # WHAT: CosineAnnealingLR의 초기화 메서드입니다.
    # WHY: 주기, 최소 학습률 등 코사인 스케줄링을 위한 환경을 준비하기 위함입니다.
    # HOW: 입력받은 인자들을 속성으로 저장합니다.
    def __init__(self, optimizer, T_max, eta_min=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = 0
        # WHAT: 스케줄러 시작 시점의 기준 학습률입니다.
        # WHY: 코사인 수식에서 최대 진폭의 기준점으로 쓰기 위함입니다.
        # HOW: 옵티마이저의 현재 lr을 저장해둡니다.
        self.base_lr = optimizer.lr
        
    # WHAT: 에포크마다 코사인 수식에 따라 학습률을 업데이트하는 메서드입니다.
    # WHY: 곡선의 다음 지점에 해당하는 학습률 값을 반영하기 위함입니다.
    # HOW: math.cos를 사용해 반환된 값을 옵티마이저 lr에 대입합니다.
    def step(self, metrics=None):
        self.last_epoch += 1
        import math
        self.optimizer.lr = self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2

# WHAT: 검증(validation) 평가 지표(metrics)가 정체될 때 학습률을 낮추는 스케줄러입니다.
# WHY: 손실값이 더 이상 떨어지지 않는 고원(plateau)에 도달했을 때 미세 조정을 유도하기 위함입니다.
# HOW: 정해진 횟수(patience) 동안 최고 기록(best)이 갱신되지 않으면 lr을 factor배 축소시킵니다.
class ReduceLROnPlateau:
    # WHAT: ReduceLROnPlateau 초기화 메서드입니다.
    # WHY: 모니터링 방식(최소화/최대화), 감소 배율, 참을성 횟수를 설정하기 위함입니다.
    # HOW: 초기 인자들을 저장하고, best 점수와 정체 카운터를 초기화합니다.
    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, min_lr=0):
        self.optimizer = optimizer
        
        # WHAT: 지표의 개선 방향입니다 ('min'은 감소, 'max'는 증가가 개선).
        # WHY: Loss는 낮아져야 좋고, Accuracy는 높아져야 좋으므로 유연성을 주기 위함입니다.
        # HOW: 문자열로 모드를 기록합니다.
        self.mode = mode
        
        # WHAT: 학습률 축소 비율입니다.
        # WHY: 정체 시 얼마나 학습률을 줄일지 강도를 결정하기 위함입니다.
        # HOW: 저장해두었다가 lr * factor에 사용합니다.
        self.factor = factor
        
        # WHAT: 참을성(유예 기간) 횟수입니다.
        # WHY: 일시적인 정체에 바로 학습률을 깎는 것을 방지하기 위함입니다.
        # HOW: 나쁜 에포크 카운트가 이를 초과할 때만 축소가 일어납니다.
        self.patience = patience
        
        # WHAT: 허용되는 최소 학습률의 하한선입니다.
        # WHY: 학습률이 너무 0에 가까워져 학습이 아예 멈추는 것을 막기 위함입니다.
        # HOW: 축소 후 이 값과 비교하여 큰 값을 취합니다(max).
        self.min_lr = min_lr
        
        # WHAT: 모니터링 중인 지표의 역대 최고 기록입니다.
        # WHY: 현재 지표가 이전보다 개선되었는지 판별하는 기준선으로 쓰기 위함입니다.
        # HOW: 초기엔 None으로 두고 첫 스텝에 설정합니다.
        self.best = None
        
        # WHAT: 지표가 개선되지 않은 연속 에포크 횟수입니다.
        # WHY: patience를 넘었는지 체크하기 위함입니다.
        # HOW: 개선되면 0으로 리셋, 아니면 1씩 증가합니다.
        self.num_bad_epochs = 0
        
    # WHAT: 매 에포크의 평가 지표를 받아 판단 후 학습률을 조절하는 메서드입니다.
    # WHY: 실시간 성능 추이를 기반으로 동적 스케줄링을 하기 위함입니다.
    # HOW: 지표 개선 여부를 확인하고 카운터를 관리하며, 조건 만족 시 옵티마이저 lr을 줄입니다.
    def step(self, metrics):
        if self.best is None:
            # WHAT: 초기 상태 설정입니다.
            # WHY: 비교할 대상이 없으므로 첫 평가 지표를 최고 기록으로 삼기 위함입니다.
            # HOW: best 변수에 metrics를 복사하고 리턴합니다.
            self.best = metrics
            return
            
        # WHAT: 지표가 이전 기록보다 나아졌는지를 나타내는 플래그 변수입니다.
        # WHY: 조건 분기를 통합하기 위함입니다.
        # HOW: mode에 따라 비교 연산을 다르게 수행합니다.
        is_better = False
        if self.mode == 'min' and metrics < self.best:
            is_better = True
        elif self.mode == 'max' and metrics > self.best:
            is_better = True
            
        if is_better:
            # WHAT: 개선된 경우의 상태 갱신입니다.
            # WHY: 최고 기록을 갱신하고 정체 상태를 초기화하기 위함입니다.
            # HOW: best를 교체하고 num_bad_epochs를 0으로 만듭니다.
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            # WHAT: 개선되지 않은 경우의 상태 갱신입니다.
            # WHY: 정체 기간을 카운트하기 위함입니다.
            # HOW: num_bad_epochs에 1을 더합니다.
            self.num_bad_epochs += 1
            
        if self.num_bad_epochs >= self.patience:
            # WHAT: 정체기가 한계를 초과했을 때 실제 학습률을 감소시키는 부분입니다.
            # WHY: 고원을 벗어나기 위해 더 세밀한 보폭을 적용하기 위함입니다.
            # HOW: 기존 학습률에 factor를 곱하고 하한선과 비교한 뒤 적용합니다.
            self.optimizer.lr = max(self.optimizer.lr * self.factor, self.min_lr)
            # WHAT: 감소 이후 정체 카운터를 리셋합니다.
            # WHY: 새로운 학습률 스케일에서 다시 patience만큼 기회를 주기 위함입니다.
            # HOW: 0으로 초기화합니다.
            self.num_bad_epochs = 0


