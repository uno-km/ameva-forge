import json
import os
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

print("1. Fetching MNIST dataset...")
# MNIST 784 픽셀 데이터셋 다운로드
mnist = fetch_openml('mnist_784', version=1, cache=True, parser='auto')
X = mnist.data.astype('float32') / 255.0  # 0~1 정규화
y = mnist.target.astype('int')

# 시간 절약을 위해 일부 샘플만 사용하거나 전체 사용 (전체 7만장)
# 로컬 테스트용이므로 60,000장만 사용해서 빠르게 학습
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=10000, random_state=42)

print("2. Training 2-Layer MLP (784 -> 128 -> 10)...")
mlp = MLPClassifier(
    hidden_layer_sizes=(128,),
    activation='relu',
    max_iter=30,      # 빠르게 데모용으로 30 에포크만
    alpha=1e-4,
    solver='adam',
    verbose=10,
    random_state=42,
    learning_rate_init=0.001
)

mlp.fit(X_train, y_train)
score = mlp.score(X_test, y_test)
print(f"3. Training complete! Test Accuracy: {score * 100:.2f}%")

print("4. Exporting weights to JSON...")
# mlp.coefs_ 는 리스트: [ (784, 128), (128, 10) ]
# mlp.intercepts_ 는 리스트: [ (128,), (10,) ]

weights = {
    "W1": mlp.coefs_[0].astype(np.float32).tolist(),
    "b1": mlp.intercepts_[0].astype(np.float32).tolist(),
    "W2": mlp.coefs_[1].astype(np.float32).tolist(),
    "b2": mlp.intercepts_[1].astype(np.float32).tolist(),
}

out_path = os.path.join(os.path.dirname(__file__), '..', 'mnist_weights.json')
with open(out_path, 'w') as f:
    json.dump(weights, f)

print(f"5. Saved successfully to {out_path} (Size: {os.path.getsize(out_path) / 1024:.1f} KB)")
