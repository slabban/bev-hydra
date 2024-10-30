import os
import hydra

from omegaconf import DictConfig
import rootutils
from lightning.pytorch.loggers import Logger


from lightning import LightningDataModule, LightningModule, Trainer, Callback

from src.models.damp import Damp

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)

import cv2
import numpy as np
import torch
import torchvision
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

from src.trainer.trainer import BevLightingModule
from src.utils.network import NormalizeInverse
from src.utils.instance import predict_instance_segmentation_and_trajectories
from src.utils.visualisation import plot_instance_map, generate_instance_colours, make_contour, convert_figure_numpy
from typing import Any, Dict, List, Optional, Tuple


rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def evaluate(cfg: DictConfig):
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data, common=cfg.common)

    log.info(f"Instantiating training module <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, common_cfg=cfg.common)

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
            "cfg": cfg,
            "datamodule": datamodule,
            "model": model,
            "logger": logger,
            "trainer": trainer,
        }
    
    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.validate(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict





if __name__ == '__main__':
    evaluate()