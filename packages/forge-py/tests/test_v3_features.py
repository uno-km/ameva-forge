import unittest
import sys
import os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import forge as at
import forge.nn as nn

class TestV3Features(unittest.TestCase):
    def test_cnn_forward_backward(self):
        print("Testing CNN...")
        # (N, C, H, W)
        x = at.tensor(np.random.randn(2, 3, 16, 16).astype(np.float32), requires_grad=True)
        
        class CNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
                self.bn = nn.BatchNorm2d(8)
                self.relu = nn.ReLU()
                self.pool = nn.MaxPool2d(2, 2)
                self.flatten = nn.Flatten()
                self.fc = nn.Linear(8 * 8 * 8, 2)
            
            def forward(self, x):
                return self.fc(self.flatten(self.pool(self.relu(self.bn(self.conv(x))))))
                
        model = CNN()
        model.train()
        out = model(x)
        self.assertEqual(out.shape, (2, 2))
        loss = out.sum()
        loss.backward()
        
        self.assertIsNotNone(model.conv.weight.grad)
        self.assertIsNotNone(model.fc.weight.grad)
        print("CNN OK")
        
    def test_rnn_forward(self):
        print("Testing Sequence Models...")
        embed = nn.Embedding(100, 16)
        # Sequence of indices (N=2, L=5)
        indices = at.tensor(np.random.randint(0, 100, (2, 5)).astype(np.float32), requires_grad=False)
        x = embed(indices)
        self.assertEqual(x.shape, (2, 5, 16))
        
        lstm = nn.LSTM(16, 32, batch_first=True)
        out, (h_n, c_n) = lstm(x)
        self.assertEqual(out.shape, (2, 5, 32))
        self.assertEqual(h_n.shape, (2, 32))
        print("RNN OK")
        
    def test_transformer_forward(self):
        print("Testing Transformer...")
        # (N, L, E)
        x = at.tensor(np.random.randn(2, 10, 64).astype(np.float32), requires_grad=True)
        layer = nn.TransformerEncoderLayer(d_model=64, nhead=4)
        out = layer(x)
        self.assertEqual(out.shape, (2, 10, 64))
        
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        print("Transformer OK")

if __name__ == '__main__':
    unittest.main()
