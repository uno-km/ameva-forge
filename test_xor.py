import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, 'packages/forge-py/src')
import numpy as np
import forge as at
from forge import nn, optim
from forge.functional import mse_loss

# XOR dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
Y = np.array([[0],[1],[1],[0]], dtype=np.float32)

# Model: 2 -> 4 -> 1
np.random.seed(42)
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)

optimizer = optim.Adam(model.parameters(), lr=0.1)

for epoch in range(1000):
    x = at.tensor(X)
    y = at.tensor(Y)
    
    pred = model(x)
    loss = mse_loss(pred, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        loss_val = loss.numpy()
        print(f'Epoch {epoch}: loss = {loss_val:.6f}')

print('Final predictions:')
final = model(at.tensor(X))
print(final.numpy())
print('XOR TRAINING TEST COMPLETE')
