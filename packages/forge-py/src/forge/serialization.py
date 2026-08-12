import numpy as np
from .nn import Module

def save_model(model: Module, path: str):
    state_dict = model.state_dict(keep_vars=False)
    numpy_dict = {}
    for k, v in state_dict.items():
        if hasattr(v, 'numpy'):
            numpy_dict[k] = v.numpy()
        else:
            numpy_dict[k] = v
            
    np.savez(path, **numpy_dict)

def load_model(model: Module, path: str):
    data = np.load(path)
    state_dict = {k: data[k] for k in data.files}
    model.load_state_dict(state_dict)
