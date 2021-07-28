import collections
import logging
import torch
import copy
import random
import numpy as np

from naslib.optimizers.core.metaclasses import MetaOptimizer

from naslib.search_spaces.core.query_metrics import Metric
from naslib.search_spaces.nasbench201.graph import NasBench201SearchSpace

from naslib.utils.utils import AttrDict, count_parameters_in_MB
from naslib.utils.logging import log_every_n_seconds

logger = logging.getLogger(__name__)


class LocalSearch(MetaOptimizer):

    # training the models is not implemented
    using_step_function = False

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.epochs = config.search.epochs

        self.performance_metric = Metric.VAL_ACCURACY
        self.dataset = config.dataset

        self.num_init = config.search.num_init
        self.nbhd = []
        self.chosen = None
        self.best_arch = None

        self.history = torch.nn.ModuleList()

    def adapt_search_space(self, search_space, scope=None, dataset_api=None):
        assert (
            search_space.QUERYABLE
        ), "Local search is currently only implemented for benchmarks."
        self.search_space = search_space.clone()
        self.scope = scope if scope else search_space.OPTIMIZER_SCOPE
        self.dataset_api = dataset_api

    def new_epoch(self, epoch):

        if epoch < self.num_init:
            # randomly sample initial architectures
            model = (
                torch.nn.Module()
            )  # hacky way to get arch and accuracy checkpointable
            model.arch = self.search_space.clone()
            model.arch.sample_random_architecture(dataset_api=self.dataset_api)
            model.accuracy = model.arch.query(
                self.performance_metric, self.dataset, dataset_api=self.dataset_api
            )

            if not self.best_arch or model.accuracy > self.best_arch.accuracy:
                self.best_arch = model
            self._update_history(model)

        else:
            if (
                len(self.nbhd) == 0
                and self.chosen
                and self.best_arch.accuracy <= self.chosen.accuracy
            ):
                logger.info(
                    "Reached local minimum. Starting from new random architecture."
                )

                model = (
                    torch.nn.Module()
                )  # hacky way to get arch and accuracy checkpointable
                model.arch = self.search_space.clone()
                model.arch.sample_random_architecture(dataset_api=self.dataset_api)
                model.accuracy = model.arch.query(
                    self.performance_metric, self.dataset, dataset_api=self.dataset_api
                )

                self.chosen = model
                self.best_arch = model
                self.nbhd = self.chosen.arch.get_nbhd(dataset_api=self.dataset_api)

            else:
                if len(self.nbhd) == 0:
                    logger.info(
                        "Start a new iteration. Pick the best architecture and evaluate its neighbors."
                    )
                    self.chosen = self.best_arch
                    self.nbhd = self.chosen.arch.get_nbhd(dataset_api=self.dataset_api)

                model = self.nbhd.pop()
                model.accuracy = model.arch.query(
                    self.performance_metric, self.dataset, dataset_api=self.dataset_api
                )

                if model.accuracy > self.best_arch.accuracy:
                    self.best_arch = model
                    logger.info("Found new best architecture.")
                self._update_history(model)

    def _update_history(self, child):
        if len(self.history) < 100:
            self.history.append(child)
        else:
            for i, p in enumerate(self.history):
                if child.accuracy > p.accuracy:
                    self.history[i] = child
                    break

    def train_statistics(self):
        best_arch = self.get_final_architecture()
        return (
            best_arch.query(
                Metric.TRAIN_ACCURACY, self.dataset, dataset_api=self.dataset_api
            ),
            best_arch.query(
                Metric.VAL_ACCURACY, self.dataset, dataset_api=self.dataset_api
            ),
            best_arch.query(
                Metric.TEST_ACCURACY, self.dataset, dataset_api=self.dataset_api
            ),
        )

    def test_statistics(self):
        best_arch = self.get_final_architecture()
        return best_arch.query(Metric.RAW, self.dataset, dataset_api=self.dataset_api)

    def get_final_architecture(self):
        return max(self.history, key=lambda x: x.accuracy).arch

    def get_op_optimizer(self):
        raise NotImplementedError()

    def get_checkpointables(self):
        return {"model": self.history}

    def get_model_size(self):
        return count_parameters_in_MB(self.history)
