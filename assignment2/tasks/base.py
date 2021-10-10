import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Union

from utils.grid import GridTable, serialize_grid_table, parse_grid_table, \
    summarize_grid_table, serialize_grid_optimization_summary, GridSummary, \
    GridOptimizationSummary, GridNNSummary


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

    @property
    @abstractmethod
    def grid_table_json(self) -> str:
        """Path to the grid table JSON file"""
        pass

    @abstractmethod
    def grid_run(self) -> GridTable:
        """Runs the grid search. All necessary parameters
        should be provided in the constructor and accessed
        here."""
        pass

    @abstractmethod
    def summarize_grid_table(self, grid_table: GridTable):
        """Summarizes a grid table"""
        pass

    @property
    @abstractmethod
    def grid_summary_json_path(self):
        """Path to grid summary JSON file"""
        pass

    @abstractmethod
    def serialize_grid_summary(self,
                               grid_summary: GridSummary) -> Dict[str, Any]:
        """Serializes a GridSummary object to a
        JSON-serializable dictionary"""
        pass

    @abstractmethod
    def parse_grid_summary(
            self, grid_summary_serialized: Dict[str, Any]) -> GridSummary:
        """Parses a serialized grid summary to a
        GridSummary object"""
        pass

    @abstractmethod
    def sync_parameter_plots(self, grid_summary: GridSummary):
        """Synchronizes parameter plots. To synchronize
        means generating the plots if they don't exist yet.
        """
        pass
