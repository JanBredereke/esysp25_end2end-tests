import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class DatasetPair:
    def __init__(self, train_set: Dataset, test_set: Dataset, validate_set: Dataset, classes: dict, batch_size: int=32, num_workers: int=0, seed: int=0):
        self.seed = seed

        self.train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=False, # for determinism
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.seed)
        )

        self.validate_loader = DataLoader(
            validate_set,
            batch_size=batch_size,
            shuffle=False, # for determinism
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.seed)
        )

        self.test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False, # for determinism
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.seed)
        )

        self.classes = classes
