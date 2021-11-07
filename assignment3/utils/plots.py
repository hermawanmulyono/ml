from typing import List, Iterable, Callable

import numpy as np
from plotly import graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler


def visualize_3d_data(x_data: np.ndarray,
                      y_data: np.ndarray,
                      categories: List[str],
                      scatter_alpha: float = 0.5) -> go.Figure:
    """Visualizes 3D data

    Args:
        x_data: A (num_examples, 3) array.
        y_data: A (num_examples, ) array of labels
            corresponding to `x_data`. categories: Category
            names of the data, ordered by indices
        categories: The category names in the dataset
        scatter_alpha: Marker alpha value. 0 is solid, 1 is
            fully transparent.

    Returns:
        A plotly figure object

    """

    fig = go.Figure()
    fig = _add_scatter_dataset3d(fig,
                                 x_data,
                                 y_data,
                                 scatter_alpha=scatter_alpha,
                                 scatter_size=4,
                                 categories=categories)
    fig.update_layout(
        scene={
            'xaxis_title': 'x1',
            'yaxis_title': 'x2',
            'zaxis_title': 'x3',
            'aspectmode': 'data'  # Fixed aspect ratio
        })

    return fig


def visualize_reduced_dataset3d(x_data: np.ndarray, y_data: np.ndarray,
                                categories: List[str]) -> go.Figure:
    """Visualizes 3D data

    Args:
        x_data: A (num_examples, n_dims) array, where
            `n_dims <= 3`.
        y_data: A (num_examples, ) array of labels
            corresponding to `x_data`. categories: Category
            names of the data, ordered by indices
        categories: The category names in the dataset

    Returns:
        A plotly figure object

    """

    fig = go.Figure()

    n_dims = x_data.shape[1]
    if n_dims < 3:
        padding = 3 - n_dims
        zeros = np.zeros((len(x_data), padding))
        x_data = np.concatenate([x_data, zeros], axis=1)

    fig = _add_scatter_dataset3d(fig,
                                 x_data,
                                 y_data,
                                 scatter_alpha=0.5,
                                 scatter_size=4,
                                 categories=categories)

    fig.update_layout(
        scene={
            'xaxis_title': 'x1',
            'yaxis_title': 'x2',
            'zaxis_title': 'x3',
            'aspectmode': 'data'  # Fixed aspect ratio
        })

    return fig


def _add_scatter_dataset3d(fig: go.Figure, x_data, y_data, scatter_alpha: float,
                           scatter_size: int, categories: List[str]):
    """Adds Dataset2D scatter plot

    Args:
        fig: A Figure object
        x_data: Dataset3D features
        y_data: Dataset3D labels
        scatter_alpha: Transparency
        scatter_size: Scatter size
        categories: Category names of the data, ordered by indices

    Returns:
        A Figure object with scatter


    """
    possible_labels = sorted(set(y_data))
    if not ((0 <= min(possible_labels)) and
            (max(possible_labels) < len(categories))):
        raise ValueError(
            f'Possible labels are not consistent with the labels {categories}')

    if x_data.shape[1] != 3:
        raise ValueError('Invalid number of dimensions')

    for label in possible_labels:
        category_name = categories[label]

        selected_indices = (y_data == label)
        x_selected = x_data[selected_indices]

        fig.add_trace(
            go.Scatter3d(x=x_selected[:, 0],
                         y=x_selected[:, 1],
                         z=x_selected[:, 2],
                         mode='markers',
                         name=category_name,
                         marker={
                             'size': scatter_size,
                             'opacity': 1 - scatter_alpha
                         }))
    return fig


def simple_line_plot(x, y, x_title: str, y_title: str, error=None) -> go.Figure:
    """Generates a simple line plot

    Args:
        x: x-axis data
        y: y-axis data
        x_title: x-axis title
        y_title: y-axis title
        error: Optional error vector. If given, continuous
            error bar will be added.

    Returns:
        A go.Figure object with corresponding x and y data.

    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, line={'color': 'rgb(0,100,80)'}))

    if error is not None:
        error_np = np.array(error)
        x_np = np.array(x)
        y_np = np.array(y)

        # x, then x reversed
        xx = np.concatenate([x_np, x_np[::-1]], axis=0)

        # upper then lower
        yy = np.concatenate([y_np + error_np, y_np[::-1] - error_np[::-1]],
                            axis=0)

        fig.add_trace(
            go.Scatter(x=xx,
                       y=yy,
                       fill='toself',
                       fillcolor='rgba(0,100,80,0.2)',
                       line=dict(color='rgba(255,255,255,0)'),
                       hoverinfo="skip",
                       showlegend=False))

    fig.update_layout({'xaxis_title': x_title, 'yaxis_title': y_title})
    return fig


class MatplotlibAdapter:
    """Adapter for a Matplotlib figure to plotly go.Figure()

    Matplotlib offers low-level operations for advanced
    figure manipulations. However, most of the time,
    it is faster to work with plotly. This class can be used
    with code that expects plotly's `write_image()`.

    """

    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        plt.close()

    def write_image(self, path_to_image):
        self.fig.savefig(path_to_image)


def visualize_fashion_mnist(x_data: np.ndarray, y_data: np.ndarray,
                            categories: List[str]):
    """Visualizes Fashion-Mnist dataset

    The t-SNE algorithm is used to map a the 784-dimensional
    vectors into 2-dimensional data.

    Some thumbnails will be displayed, along with the
    t-SNE projection.

    The Matplotlib visualization is adapted from
    https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html

    Args:
        x_data: Fashion-MNIST feature array
        y_data: The `x_data` labels
        categories: An ordered list of category names

    Returns:
        A MatplotlibAdapter object, which has a
            `write_image()` method, just like go.Figure.
    """

    if len(x_data) != len(y_data):
        raise ValueError('x_data and y_data must have the same length.')

    if x_data.shape[1] != 784:
        raise ValueError('x_data of Fashion-MNIST must be 784-dimensional')

    return _tsne_fashion_mnist_visualization(x_data, y_data, x_data, categories)


def get_visualize_reduced_dataset3d_fn(x_original: np.ndarray):
    if x_original.shape[1] != 3:
        raise ValueError('x_data of Dataset3D must be 784-dimensional')

    len_data = len(x_original)

    def visualize_reduced_dataset3d_fn(x_data: np.ndarray, y_data: np.ndarray,
                                       categories: List[str]):

        if len(x_data) != len_data:
            raise ValueError(f'Expecting data of length {len_data}, but got'
                             f'{len(x_data)}')

        # With original space
        fig1 = visualize_3d_data(x_original, y_data, categories)

        # With reduced space
        fig2 = visualize_reduced_dataset3d(x_data, y_data, categories)

        return fig1, fig2

    return visualize_reduced_dataset3d_fn


def get_visualize_reduced_fashion_mnist_fn(x_original: np.ndarray):
    """Gets a function to visualize reduced Fashion-MNIST

    A reduced Fashion-MNIST dataset is Fashion-MNIST after
    a dimensionality reduction algorithm.

    To visualize it, the t-SNE algorithm is used to map the
    reduced data, `x_data` to a 2D plane. Then, the
    visualization contains some thumbnails of the original
    data.

    Args:
        x_original: The original features of shape
            `(N, n_original_features)`. Note that
            `n_original_features = 784`

    Returns:
        A function which takes `x_data, y_data, categories`
            and returns a `go.Figure` object.

    """

    if x_original.shape[1] != 784:
        raise ValueError('x_data of Fashion-MNIST must be 784-dimensional')

    len_data = len(x_original)

    def visualize_reduced_fashion_mnist(x_data: np.ndarray, y_data: np.ndarray,
                                        categories: List[str]):

        if len(x_data) != len_data:
            raise ValueError(f'Expecting data of length {len_data}, but got'
                             f'{len(x_data)}')

        # With original space
        fig1 = _tsne_fashion_mnist_visualization(x_original, y_data, x_original,
                                                 categories)

        # With reduced space
        fig2 = _tsne_fashion_mnist_visualization(x_data, y_data, x_original,
                                                 categories)
        return fig1, fig2

    return visualize_reduced_fashion_mnist


def _tsne_fashion_mnist_visualization(x_data: np.ndarray, y_data: np.ndarray,
                                      x_original: np.ndarray,
                                      categories: List[str]):
    """Implementation of Fashion-MNIST t-SNE visualization,
    but the interface is not compatible with the
    visualization function interface.
    """

    tsne = TSNE()
    transformed = tsne.fit_transform(x_data)
    assert len(transformed.shape) == 2

    fig, ax = plt.subplots()
    fig.set_size_inches(10, 10)
    transformed = MinMaxScaler().fit_transform(transformed)

    possible_labels = sorted(set(y_data))
    if not ((0 <= min(possible_labels)) and
            (max(possible_labels) < len(categories))):
        raise ValueError(
            f'Possible labels are not consistent with the labels {categories}')

    original_indices = np.arange(len(x_data))

    shown_images = np.array([[1.0, 1.0]])  # initially, just something big

    for label in possible_labels:
        category_name = categories[label]

        selected_indices = original_indices[y_data == label]
        x_selected = transformed[selected_indices]

        plt.scatter(x_selected[:, 0], x_selected[:, 1], label=category_name)

        for i in selected_indices:
            # show an annotation box for a group of digits
            dist = np.sum((transformed[i] - shown_images)**2, 1)
            if np.min(dist) < 6e-3:
                # don't show points that are too close
                continue
            shown_images = np.concatenate([shown_images, [transformed[i]]],
                                          axis=0)

            x_disp = x_original[i]
            thumbnail = np.reshape(x_disp, (28, 28))

            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(thumbnail, cmap=plt.cm.gray_r),
                transformed[i])
            ax.add_artist(imagebox)

    ax.legend()
    adapter_figure = MatplotlibAdapter(fig, ax)

    return adapter_figure


def feature_importance_chart(feature_importances: np.ndarray):
    """Feature importance bar chart

    The bar plot shows the feature importance scores,
    sorted from the highest to lowest.

    Args:
        feature_importances:

    Returns:
        A go.Figure() object containing the bar chart.

    """

    sorted_importances = -np.sort(-feature_importances)
    x = np.arange(0, len(feature_importances))

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=sorted_importances))
    return fig


def visualize_dataset3d_vectors(vectors: np.ndarray, x_data: np.ndarray,
                                y_data: np.ndarray):

    _validate_vector_visualization_inputs(vectors, x_data, y_data)

    if x_data.shape[1] != 3:
        raise ValueError('Expecting 3-dimensional x_data')

    # Calculate the length of vectors in figure
    scale = np.max(np.max(x_data, axis=0) - np.min(x_data, axis=0))

    categories = sorted(set(y_data))
    fig = visualize_3d_data(x_data,
                            y_data, [f'{c}' for c in categories],
                            scatter_alpha=0.9)

    x_mean = x_data.mean(axis=0)

    for vector in vectors:
        vector = vector / np.linalg.norm(vector) * scale
        x_arrow = np.array([0, vector[0]]) + x_mean[0]
        y_arrow = np.array([0, vector[1]]) + x_mean[1]
        z_arrow = np.array([0, vector[2]]) + x_mean[2]
        fig.add_trace(
            go.Scatter3d(x=x_arrow,
                         y=y_arrow,
                         z=z_arrow,
                         mode='lines',
                         line={'width': 5}))

    fig.update_layout(
        scene={
            'xaxis_title': 'x1',
            'yaxis_title': 'x2',
            'zaxis_title': 'x3',
            'aspectmode': 'data'  # Fixed aspect ratio
        })

    return fig


def visualize_fashionmnist_vectors(vectors: np.ndarray, x_data: np.ndarray,
                                   y_data: np.ndarray):

    _validate_vector_visualization_inputs(vectors, x_data, y_data)

    n_dims = x_data.shape[1]
    thumbnail_length = 28
    expected_n_dims = thumbnail_length**2
    if n_dims != expected_n_dims:
        raise ValueError('Wrong number of features. Expecting '
                         f'{expected_n_dims}, but got {n_dims}')

    # Just pick at most 64 vectors
    vectors = vectors[:64]

    # Rescale all vectors to 0-255
    min_vectors = np.min(vectors, axis=1)
    max_vectors = np.max(vectors, axis=1)
    numerator = vectors - min_vectors.reshape((-1, 1))
    denominator = max_vectors - min_vectors
    rescaled_vectors = (numerator / denominator.reshape(
        (-1, 1)) * 255).astype(np.uint8)

    # Use m x m subplot to include the vectors as thumbnails
    m = np.ceil(np.sqrt(len(vectors)))

    # Gap between images
    gap = 10

    canvas_size = int((thumbnail_length + gap) * m + gap)
    canvas = np.zeros((canvas_size, canvas_size, 3))
    canvas[:, :] = [255, 255, 0]  # Yellow background

    for i, vector in enumerate(rescaled_vectors):
        row = int((i // m) * (gap + thumbnail_length) + gap)
        col = int(i % m * (gap + thumbnail_length) + gap)

        reshaped = np.reshape(vector, (thumbnail_length, thumbnail_length))
        stacked = np.stack([reshaped] * 3, axis=-1)

        canvas[row:row + thumbnail_length, col:col + thumbnail_length] = stacked

    fig = px.imshow(canvas)
    fig.update_layout(xaxis_visible=False,
                      xaxis_showticklabels=False,
                      yaxis_visible=False,
                      yaxis_showticklabels=False)

    # fig.show()
    return fig


def _validate_vector_visualization_inputs(vectors: np.ndarray,
                                          x_data: np.ndarray,
                                          y_data: np.ndarray):
    if len(x_data) != len(y_data):
        raise ValueError('x_data and y_data must have the same length.')

    if len(x_data.shape) != 2:
        raise ValueError('x_data must be a 2-dimensional array')

    if len(vectors.shape) != 2:
        raise ValueError('vectors must be a 2-dimensional array')

    if vectors.shape[1] != x_data.shape[1]:
        raise ValueError('vectors and x_data must have the same number of '
                         'features')
