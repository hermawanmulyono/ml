import copy
import logging
from typing import Tuple, Optional, Dict, List

import progressbar
import torch
import torch.nn.functional as F
import numpy as np
import plotly.graph_objects as go


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


def focal_loss_fn(y_pred: torch.Tensor,
                  y_target: torch.Tensor,
                  gamma: float = 0.0,
                  alpha: float = 1.0):

    if y_pred.shape[0] != y_target.shape[0]:
        raise ValueError(f'Invalid input shapes')

    num_classes = y_pred.shape[1]

    target_one_hot = F.one_hot(y_target, num_classes)

    pt_gamma = torch.pow(1 - y_pred, gamma)

    alphas = alpha * target_one_hot + (1 - alpha) * (1 - target_one_hot)

    losses = -alphas * pt_gamma * torch.log(y_pred)

    losses_sum = torch.sum(losses, dim=1)
    losses_mean = torch.mean(losses_sum, dim=0)

    return losses_mean


def cross_entropy_fn(y_pred: torch.Tensor, y_target: torch.Tensor):
    log_likelihood = torch.log(y_pred)
    loss = F.nll_loss(log_likelihood, y_target)
    return loss


def generate_batches(x_data: np.ndarray, y_data: np.ndarray, batch_size: int,
                     trailing: bool):
    """Generates data in batches

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
                 hidden_layers: List[int]):

        assert hidden_layers

        hidden_layers_tensors = [torch.nn.BatchNorm1d(num_features=in_features)]
        for i, hidden_layer_size in enumerate(hidden_layers):
            if i == 0:
                in_features_ = in_features
            else:
                in_features_ = hidden_layers[i - 1]

            hidden_layers_tensors.append(
                torch.nn.Linear(in_features_, hidden_layer_size))
            hidden_layers_tensors.append(torch.nn.ReLU())

        output_layer = [
            torch.nn.Linear(hidden_layers[-1], num_classes),
            torch.nn.Softmax(dim=1)
        ]

        layers = hidden_layers_tensors + output_layer

        model = torch.nn.Sequential(*layers)

        self.model = model
        self.num_classes = num_classes
        self.device = torch.device(
            "cuda:0" if not torch.cuda.is_available() else "cpu")
        self._training_log: Optional[Dict[str, list]] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

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

        logging.info('Training neural network')

        device = self.device
        self.model.to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.optimizer = optimizer

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

                loss = cross_entropy_fn(y_pred, y_tensor)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self._update_training_log(x_train, y_train, x_eval, y_eval)

        if verbose:
            bar.finish()

    def _update_training_log(self, x_train, y_train, x_eval, y_eval):
        if self._training_log is None:
            self._training_log = {
                'epoch': [],
                'train_loss': [],
                'eval_loss': [],
                'train_accuracy': [],
                'eval_accuracy': []
            }

        training_log = self._training_log
        epoch = len(self._training_log['epoch']) + 1
        training_log['epoch'].append(epoch)

        device = self.device

        self.model.to(self.device)

        def _update(x_data: np.ndarray, y_data: np.ndarray, set_: str):

            assert set_ in {'train', 'eval'}

            with torch.no_grad():
                x_tensor: torch.Tensor = torch.Tensor(x_data).to(device)
                y_pred = self.model(x_tensor)
                y_tensor: torch.Tensor = torch.Tensor(y_data).type(
                    torch.long).to(device)

                loss_tensor: torch.Tensor = cross_entropy_fn(y_pred, y_tensor)
                loss_np = loss_tensor.cpu().numpy()
                training_log[f'{set_}_loss'].append(float(loss_np))

                labels_pred: torch.Tensor = torch.argmax(y_pred, dim=1)
                acc = torch.sum(labels_pred == y_tensor) / len(x_data)
                acc_float = float(acc.cpu().numpy())
                training_log[f'{set_}_accuracy'].append(acc_float)

        # Training set
        _update(x_train, y_train, 'train')

        # Evaluation set
        if x_eval is not None and y_eval is not None:
            _update(x_eval, y_eval, 'eval')

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
        self._training_log = state_dict['training_log']

    def save(self, path_to_state_dict: str):
        model_state_dict = self.model.state_dict()
        training_log_dict = self._training_log
        state_dict = {
            'model': model_state_dict,
            'training_log': training_log_dict
        }
        torch.save(state_dict, path_to_state_dict)

    @property
    def training_log(self):
        """Training log of the neural network"""
        d = copy.deepcopy(self._training_log)

        return d


def training_curves(training_log: Dict[str, list]):
    """Generates training curves from the training log

    There are two curves:
      - Loss curves
      - Accuracy curves

    Args:
        training_log: A dictionary with the following structure

           `{'epoch': ..., 'train_loss': ..., 'eval_loss': ...,
           'train_accuracy': ..., 'eval_accuracy': ...}`

    Returns:
        (loss_fig, acc_fig)

    """

    def _make_plot(train_key: str, eval_key: str, train_legend: str,
                   eval_legend: str, y_axis_title: str, fig_title: str):

        fig = go.Figure()

        x = training_log['epoch']
        train_loss = training_log[train_key]
        eval_loss = training_log[eval_key]

        fig.add_trace(
            go.Scatter(x=x, y=train_loss, mode='lines', name=train_legend))
        fig.add_trace(
            go.Scatter(x=x, y=eval_loss, mode='lines', name=eval_legend))

        fig.update_layout({
            'xaxis_title': 'epoch',
            'title_x': 0.5,
            'yaxis_title': y_axis_title,
            'title': fig_title
        })

        return fig

    loss_fig = _make_plot('train_loss', 'eval_loss', 'Train loss', 'Eval loss',
                          'Cross-entropy loss',
                          'Training and Evaluation Loss Curves')
    acc_fig = _make_plot('train_accuracy', 'eval_accuracy', 'Train accuracy',
                         'Eval accuracy', 'Accuracy',
                         'Training and Evaluation Accuracy Curves')

    return loss_fig, acc_fig
