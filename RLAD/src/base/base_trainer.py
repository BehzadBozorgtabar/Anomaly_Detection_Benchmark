from abc import ABC, abstractmethod
from .base_dataset import BaseADDataset
from .base_net import BaseNet


class BaseTrainer(ABC):
    """Trainer base class."""

    def __init__(self, lr: float, n_epochs: int, batch_size: int, rep_dim: int, K : int,
                 weight_decay: float, device: str, n_jobs_dataloader: int, w_rec:float, w_contrast):
        super().__init__()
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rep_dim = rep_dim
        self.K = K
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader
        self.w_rec = w_rec
        self.w_contrast = w_contrast

    @abstractmethod
    def train(self) -> BaseNet:
        """
        Implement train method that trains the given network using the train_set of dataset.
        :return: Trained net
        """
        pass

    @abstractmethod
    def test(self) -> float:
        """
        Implement test method that evaluates the test_set of dataset on the given network.
        """
        pass
