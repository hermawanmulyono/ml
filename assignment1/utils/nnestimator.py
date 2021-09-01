from typing import Tuple, Optional, Dict
import os

import progressbar
import torch
import numpy as np


def one_hot(y: np.ndarray, num_classes: int):
    """Converts to one-hot vector

    Args:
        y: A one dimensional numpy array of shape (N, )
        num_classes: Number of classes

    Returns:
        One hot numpy array of shape (N, nclasses)

    """
    assert len(y.shape) == 1
    assert np.all((y >= 0) & (y < num_classes))

    y_oh = np.zeros((len(y), num_classes))

    for c in range(num_classes):
        c_indices = y == c
        y_oh[c_indices, c] = 1

    return y_oh


def shuffle(x_data: np.ndarray,
            y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffles X and y vectors for training purposes

    Args:
        x_data: An array of shape (N, num_features)
        y_data: An array of shape (N, ) or (N, num_classes)

    Returns:
        Tuple (x_shuffled, y_shuffled)

    """
    assert len(x_data) == len(y_data)
    indices = np.random.permutation(len(x_data))
    x_shuffled = x_data[indices]
    y_shuffled = y_data[indices]
    return x_shuffled, y_shuffled


def generate_batches(x_data: np.ndarray, y_data: np.ndarray, batch_size: int,
                     trailing: bool):
    """

    Args:
        x_data: An array of shape (N, num_features)
        y_data: An array of shape (N, ) or (N, num_classes)
        batch_size: Batch size
        trailing: If True, the last batch will always be given. If False,
            the it is the case when the last batch has `batch_size` examples

    Yields:
        x_batch, y_batch

    """
    assert len(x_data) == len(y_data)

    for i in range(0, len(x_data), batch_size):
        x_batch = x_data[i:i + batch_size]
        y_batch = y_data[i:i + batch_size]

        if (not trailing) and (len(x_batch) != batch_size):
            continue

        yield x_batch, y_batch


class NeuralNetworkEstimator:
    """
    A class that encapsulates Pytorch's sequential fully connected model.
    This is compatible with scikit-learn's estimators and hence implement
    `fit()`, `predict()`, and `predict_proba()`

    """

    def __init__(self, in_features: int, num_classes: int,
                 num_hidden_layers: int, hidden_layer_size: int):

        assert num_hidden_layers >= 1

        hidden_layers = []
        for i in range(num_hidden_layers):
            if i == 0:
                in_features_ = in_features
            else:
                in_features_ = num_hidden_layers

            hidden_layers.append(
                torch.nn.Linear(in_features_, hidden_layer_size))
            hidden_layers.append(torch.nn.ReLU())

        output_layer = [
            torch.nn.Linear(hidden_layer_size, num_classes),
            torch.nn.Softmax(dim=1)
        ]

        layers = hidden_layers + output_layer

        model = torch.nn.Sequential(*layers)

        self.model = model
        self.num_classes = num_classes
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.training_log: Optional[Dict[str, list]] = None

    def fit(self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            x_eval: np.ndarray = None,
            y_eval: np.ndarray = None,
            learning_rate=1e-6,
            batch_size=64,
            epochs=50,
            verbose=False):

        assert len(x_train) == len(y_train)

        if verbose:
            print('Training neural network')

        device = self.device
        self.model.to(device)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

        x_train = x_train.copy()

        if verbose:
            bar = progressbar.ProgressBar(max_value=epochs)
        else:
            bar = None

        for epoch in range(epochs):
            if verbose:
                bar.update(epoch)

            x_shuffled, y_shuffled = shuffle(x_train, y_train)

            generator = generate_batches(x_shuffled,
                                         y_shuffled,
                                         batch_size,
                                         trailing=True)
            for x_batch, y_batch in generator:
                x_tensor = torch.Tensor(x_batch).to(device)
                y_pred: torch.Tensor = self.model(x_tensor)
                y_tensor = torch.Tensor(y_batch).type(torch.long).to(device)
                loss = loss_fn(y_pred, y_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self._update_training_log(x_train, y_train, x_eval, y_eval)

        if verbose:
            bar.finish()

    def _update_training_log(self, x_train, y_train, x_eval, y_eval):
        if self.training_log is None:
            self.training_log = {
                'epoch': [],
                'train_loss': [],
                'eval_loss': []
            }

        training_log = self.training_log
        epoch = len(self.training_log['epoch']) + 1
        training_log['epoch'].append(epoch)

        loss_fn = torch.nn.CrossEntropyLoss()

        self.model.to(self.device)

        with torch.no_grad():
            # Training set
            x_tensor: torch.Tensor = torch.Tensor(x_train).to(self.device)
            y_pred = self.model(x_tensor)
            y_tensor: torch.Tensor = torch.Tensor(y_train).type(torch.long).to(
                self.device)

            train_loss: torch.Tensor = loss_fn(y_pred, y_tensor)
            train_loss_np = train_loss.cpu().numpy()
            training_log['train_loss'].append(float(train_loss_np))

            # Evaluation set
            if x_eval is not None and y_eval is not None:
                x_tensor: torch.Tensor = torch.Tensor(x_eval).to(self.device)
                y_pred = self.model(x_tensor)
                y_tensor: torch.Tensor = torch.Tensor(y_eval).type(
                    torch.long).to(self.device)

                eval_loss: torch.Tensor = loss_fn(y_pred, y_tensor)
                eval_loss_np = eval_loss.cpu().numpy()
                training_log['eval_loss'].append(float(eval_loss_np))

    def predict_proba(self, x_data: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            device = self.device
            self.model.to(device)
            x_tensor = torch.Tensor(x_data).to(device)
            y_pred: torch.Tensor = self.model(x_tensor)
            y_pred_np = y_pred.cpu().numpy().copy()

        return y_pred_np

    def predict(self, x_data: np.ndarray) -> np.ndarray:
        y_pred_np = self.predict_proba(x_data)
        assert len(y_pred_np.shape) == 2, 'y_pred_np must be 2-dimensional'
        args_max = np.argmax(y_pred_np, axis=-1)
        return args_max

    def load(self, path_to_state_dict: str):
        state_dict = torch.load(path_to_state_dict)
        self.model.load_state_dict(state_dict['model'])
        self.training_log = state_dict['training_log']

    def save(self, path_to_state_dict: str):
        model_state_dict = self.model.state_dict()
        training_log_dict = self.training_log
        state_dict = {
            'model': model_state_dict,
            'training_log': training_log_dict
        }
        torch.save(state_dict, path_to_state_dict)


