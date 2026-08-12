import sys
import os
import unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'forge-py', 'src'))
import forge as at
from forge import Tensor

class TestDataLoader(unittest.TestCase):
    def test_dataloader_iteration(self):
        x = np.arange(10)
        y = np.arange(10, 20)
        dl = at.DataLoader(x, y, batch_size=3, shuffle=False)
        batches = list(dl)
        self.assertEqual(len(batches), 4)
        self.assertEqual(batches[0][0].shape, (3,))
        self.assertEqual(batches[-1][0].shape, (1,))
        
    def test_dataloader_shuffle(self):
        x = np.arange(100)
        y = np.arange(100)
        dl1 = at.DataLoader(x, y, batch_size=10, shuffle=True)
        dl2 = at.DataLoader(x, y, batch_size=10, shuffle=True)
        b1 = next(iter(dl1))[0].numpy()
        b2 = next(iter(dl2))[0].numpy()
        # High probability they are different
        self.assertFalse(np.array_equal(b1, b2))
