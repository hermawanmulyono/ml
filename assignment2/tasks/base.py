import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Union

import joblib
import numpy as np
import plotly.graph_objects as go

from utils.grid import GridTable, serialize_grid_table, parse_grid_table, \
    summarize_grid_table, serialize_grid_optimization_summary, GridSummary, \
    GridOptimizationSummary, GridNNSummary
from utils.plots import parameter_plot, fitness_curve_plot


class ExperimentBase(ABC):

    def run(self):
        json_path = self.grid_table_json

        # Only run grid-search if the JSON file doesn't exist
        if not os.path.exists(json_path):
            grid_table = self.grid_run()
            grid_table_serialized = serialize_grid_table(grid_table)

            # Write results to disk
            with open(json_path, 'w') as j:
                json.dump(grid_table_serialized, j, indent=2)

        with open(json_path, 'r') as j:
            grid_table_serialized = json.load(j)

        grid_table = parse_grid_table(grid_table_serialized)
        grid_summary = self.summarize_grid_table(grid_table)

        grid_summary_json_path = self.grid_summary_json_path

        # Only summarize results when the JSON file doesn't exist
        if not os.path.exists(grid_summary_json_path):
            with open(grid_summary_json_path, 'w') as j:
                grid_summary_serialized = \
                    self.serialize_grid_summary(grid_summary)
                json.dump(grid_summary_serialized, j, indent=2)

        # Check by opening the serialized results
        with open(grid_summary_json_path) as j:
            grid_summary_serialized = json.load(j)

        grid_summary: GridSummary = self.parse_grid_summary(
            grid_summary_serialized)

        self.sync_parameter_plots(grid_summary)
        self.sync_fitness_vs_iteration_plot(grid_summary)

    def sync_parameter_plots(self, grid_summary: GridSummary):
        """Synchronizes parameter plots.

        To synchronize means to generate the plots if
        they do not exist yet.
        """
        for metric_name in self.plot_metrics:
            for param_name, scale in self.plot_hyperparameters:
                figure_path = self.hyperparameter_plot_path(
                    metric_name, param_name)

                # Only generate plot if it doesn't exist
                if os.path.exists(figure_path):
                    continue

                fig = parameter_plot(grid_summary,
                                     param_name,
                                     scale,
                                     y_axis=metric_name)

                fig.write_image(figure_path)

    def sync_fitness_vs_iteration_plot(self, grid_summary: GridSummary):
        """Synchronizes fitness vs iteration plot

        To synchronize means to generate the plot if it
        does not exist yet.

        Args:
            grid_summary: A GridSummary object.

        Returns:
            None. This function writes an image to the
                file system.

        """

        fitness_curve_joblib_path = f'{self.fitness_curve_name}.joblib'

        # Generate the fitness curve array if necessary. This involves
        # running an experiment with the best kwargs
        fitness_curve: np.ndarray
        if not os.path.exists(fitness_curve_joblib_path):
            best_kwargs = grid_summary.kwargs

            fitness_curve = self.generate_fitness_curve(best_kwargs)
            assert len(fitness_curve.shape) == 1, \
                f'{fitness_curve.shape} is not a 1-D array'

            joblib.dump(fitness_curve, fitness_curve_joblib_path)
        else:
            fitness_curve = joblib.load(fitness_curve_joblib_path)

        fitness_curve_png_path = f'{self.fitness_curve_name}'

        # Generate the actual figure with the fitness_curve
        if not os.path.exists(fitness_curve_png_path):
            fig = fitness_curve_plot(fitness_curve)
            fig.write_image(fitness_curve_png_path)

    @property
    @abstractmethod
    def grid_table_json(self) -> str:
        """Path to the grid table JSON file"""
        raise NotImplementedError

    @abstractmethod
    def grid_run(self) -> GridTable:
        """Runs the grid search. All necessary parameters
        should be provided in the constructor and accessed
        here."""
        raise NotImplementedError

    @abstractmethod
    def summarize_grid_table(self, grid_table: GridTable):
        """Summarizes a grid table"""
        raise NotImplementedError

    @property
    @abstractmethod
    def grid_summary_json_path(self):
        """Path to grid summary JSON file"""
        raise NotImplementedError

    @abstractmethod
    def serialize_grid_summary(self,
                               grid_summary: GridSummary) -> Dict[str, Any]:
        """Serializes a GridSummary object to a
        JSON-serializable dictionary"""
        raise NotImplementedError

    @abstractmethod
    def parse_grid_summary(
            self, grid_summary_serialized: Dict[str, Any]) -> GridSummary:
        """Parses a serialized grid summary to a
        GridSummary object"""
        raise NotImplementedError

    @property
    @abstractmethod
    def plot_hyperparameters(self) -> List[Tuple[str, str]]:
        """List of [..., (hyperparameter, scale), ...]
        where `scale` is either 'linear' or 'logarithmic'.
        For each hyperparameter, the metrics defined in
        `self.plot_metrics` will be plotted.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def plot_metrics(self) -> List[str]:
        """List of metrics to plot. Each element must be a
        valid grid summary field. See the following classes
        `GridOptimizationSummary`, `GridNNSummary`.
        """
        raise NotImplementedError

    @abstractmethod
    def hyperparameter_plot_path(self, param_name: str, metric: str):
        raise NotImplementedError

    @abstractmethod
    def generate_fitness_curve(self, best_kwargs: Dict[str, Any]) -> np.ndarray:
        raise NotImplementedError

    @property
    @abstractmethod
    def fitness_curve_name(self) -> str:
        raise NotImplementedError
