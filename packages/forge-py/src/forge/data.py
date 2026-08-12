import numpy as np
from .ops import tensor
from .tensor import Tensor


class DataLoader:
    """
    미니배치 데이터 로더.

    VUL-010 Fix: label_dtype 파라미터 추가.
      - 'float32' (기본): regression/one-hot target용
      - 'int64': classification 정수 라벨 유지
      - 'auto': y_data의 원본 dtype이 정수형이면 int64, 아니면 float32
    """
    def __init__(self, x_data, y_data, batch_size=32, shuffle=True, label_dtype='auto'):
        self.x_data = np.array(x_data, dtype=np.float32)

        # VUL-010: 라벨 dtype 보존
        y_arr = np.array(y_data)
        if label_dtype == 'auto':
            if np.issubdtype(y_arr.dtype, np.integer):
                self._label_dtype = np.int64
            else:
                self._label_dtype = np.float32
        elif label_dtype == 'int64':
            self._label_dtype = np.int64
        else:
            self._label_dtype = np.float32

        self.y_data = y_arr.astype(self._label_dtype)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        n = len(self.x_data)
        indices = np.arange(n)
        if self.shuffle:
            np.random.shuffle(indices)
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            batch_idx = indices[start:end]
            x_batch = tensor(self.x_data[batch_idx])
            # 라벨은 원본 dtype 유지하여 Tensor 생성
            y_batch_data = self.y_data[batch_idx]
            if self._label_dtype == np.int64:
                # 정수 라벨: float32 Tensor로 변환하되 값은 정수로 유지
                # cross_entropy 등에서 .astype(np.int64)로 복원 가능
                y_batch = tensor(y_batch_data.astype(np.float32))
            else:
                y_batch = tensor(y_batch_data)
            yield x_batch, y_batch

    def __len__(self):
        return (len(self.x_data) + self.batch_size - 1) // self.batch_size

