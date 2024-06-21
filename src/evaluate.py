# Original Implementation
from argparse import ArgumentParser

import torch
from tqdm import tqdm

from src.data.data import prepare_dataloaders
from src.trainer.trainer import TrainingModule
from src.trainer.metrics import IntersectionOverUnion, PanopticMetric
from src.utils.network import preprocess_batch
from src.utils.instance import predict_instance_segmentation_and_trajectories

# 30mx30m, 100mx100m
EVALUATION_RANGES = {'30x30': (70, 130),
                     '100x100': (0, 200)
                     }


def eval(checkpoint_path, dataroot, version):
    trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=True)
    print(f'Loaded weights from \n {checkpoint_path}')
    trainer.eval()

    device = torch.device('cuda:0')
    trainer.to(device)
    model = trainer.model

    cfg = model.cfg
    cfg.GPUS = "[0]"
    cfg.BATCHSIZE = 1

    cfg.DATASET.DATAROOT = dataroot
    cfg.DATASET.VERSION = version

    _, valloader = prepare_dataloaders(cfg)

    panoptic_metrics = {}
    iou_metrics = {}
    n_classes = len(cfg.SEMANTIC_SEG.WEIGHTS)
    for key in EVALUATION_RANGES.keys():
        panoptic_metrics[key] = PanopticMetric(n_classes=n_classes, temporally_consistent=True).to(
            device)
        iou_metrics[key] = IntersectionOverUnion(n_classes).to(device)

    for i, batch in enumerate(tqdm(valloader)):
        preprocess_batch(batch, device)
        image = batch['image']
        intrinsics = batch['intrinsics']
        extrinsics = batch['extrinsics']
        future_egomotion = batch['future_egomotion']

        batch_size = image.shape[0]

        labels, future_distribution_inputs = trainer.prepare_future_labels(batch)

        with torch.no_grad():
            # Evaluate with mean prediction
            noise = torch.zeros((batch_size, 1, model.latent_dim), device=device)
            output = model(image, intrinsics, extrinsics, future_egomotion,
                           future_distribution_inputs, noise=noise)

        # Consistent instance seg
        pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
            output, compute_matched_centers=False, make_consistent=True
        )

        segmentation_pred = output['segmentation'].detach()
        segmentation_pred = torch.argmax(segmentation_pred, dim=2, keepdims=True)

        for key, grid in EVALUATION_RANGES.items():
            limits = slice(grid[0], grid[1])
            panoptic_metrics[key](pred_consistent_instance_seg[..., limits, limits].contiguous().detach(),
                                  labels['instance'][..., limits, limits].contiguous()
                                  )

            iou_metrics[key](segmentation_pred[..., limits, limits].contiguous(),
                             labels['segmentation'][..., limits, limits].contiguous()
                             )

    results = {}
    for key, grid in EVALUATION_RANGES.items():
        panoptic_scores = panoptic_metrics[key].compute()
        for panoptic_key, value in panoptic_scores.items():
            results[f'{panoptic_key}'] = results.get(f'{panoptic_key}', []) + [100 * value[1].item()]

        iou_scores = iou_metrics[key].compute()
        results['iou'] = results.get('iou', []) + [100 * iou_scores[1].item()]

    for panoptic_key in ['iou', 'pq', 'sq', 'rq']:
        print(panoptic_key)
        print(' & '.join([f'{x:.1f}' for x in results[panoptic_key]]))


if __name__ == '__main__':
    parser = ArgumentParser(description='Fiery evaluation')
    parser.add_argument('--checkpoint', default='./fiery.ckpt', type=str, help='path to checkpoint')
    parser.add_argument('--dataroot', default='./nuscenes', type=str, help='path to the dataset')
    parser.add_argument('--version', default='trainval', type=str, choices=['mini', 'trainval'],
                        help='dataset version')

    args = parser.parse_args()

    eval(args.checkpoint, args.dataroot, args.version)




# hydra implementation

from typing import Any, Dict, List, Tuple

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

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
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
