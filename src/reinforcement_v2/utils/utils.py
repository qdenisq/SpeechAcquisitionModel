import importlib
from gym import spaces
import numpy as np


def str_to_class(module_name, class_name):
    try:
        module_ = importlib.import_module(module_name)
        try:
            class_ = getattr(module_, class_name)
        except AttributeError:
            raise AttributeError(f'Class does not exist: {class_name}')
    except ImportError:
        raise ImportError(f'Module does not exist: {module_name}')
    return class_ or None


def convert_to_gym(space):
    return spaces.Box(np.array(space[0]), np.array(space[1]))