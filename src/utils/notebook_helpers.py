import os
import sys
import netron
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import IFrame
from collections import Counter
from torch.utils.data import Dataset

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def img_show(img):
    """
    Show training/test images
    """
    img = img / 2 + 0.5 # Unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def select_device():
    """
    Selects appropriate device for training/inference
    Will automatically select cuda if an Nvidia GPU with appropriate drivers is available
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def show_in_netron(model_filename):
    """
    Show ONNX models in netron
    """
    netron.start(model_filename, address=('0.0.0.0', 8081), browse=False)
    return IFrame(src='http://127.0.0.1:8081/', width='100%', height=400)

def set_determinism(seed: int) -> None:
    """
    Configure deterministic behavior for all applicable libraries.
    For unknown reasons, this does not make e.g. the whole training process determistic and reproducible.
    Nevertheless, any determinism is better than none.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for deterministic CUDA behavior
    # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(seed)

def print_num_samples(name: str, dataset: Dataset, classes: dict) -> None:
    """
    Print amount of samples in a dataset, total and per class
    """
    labels = [sample[1] for sample in dataset]
    class_id_to_name = { v: k for k, v in classes.items() }

    print(f"{len(dataset)} {name} samples:")
    for class_id, count in Counter(labels).items():
        print(f"  {count} {class_id_to_name[class_id]} samples")
