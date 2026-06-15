"""Entry point for UG-rPPG (EDL) training and testing."""

import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import get_config
from dataset import data_loader
from neural_methods import trainer

RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

general_generator = torch.Generator()
general_generator.manual_seed(RANDOM_SEED)
train_generator = torch.Generator()
train_generator.manual_seed(RANDOM_SEED)

DATASET_LOADERS = {
    "PURE": data_loader.PURELoader.PURELoader,
    "UBFC": data_loader.UBFCrPPGLoader.UBFCrPPGLoader,
    "MMPD": data_loader.MMPDLoader.MMPDLoader,
}


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def add_args(parser):
    parser.add_argument(
        "--config_file",
        required=False,
        default="configs/train_configs/cross/PURE_UBFC-rPPG_SIMPLEMAMBA.yaml",
        type=str,
        help="Path to experiment yaml config.",
    )
    return parser


def _resolve_loader(dataset_name):
    if dataset_name not in DATASET_LOADERS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Supported datasets: {sorted(DATASET_LOADERS.keys())}"
        )
    return DATASET_LOADERS[dataset_name]


def _build_loader(dataset_name, split_name, data_path, config_data, batch_size, shuffle, num_workers):
    if not dataset_name or not data_path:
        return None
    loader_cls = _resolve_loader(dataset_name)
    dataset = loader_cls(name=split_name, data_path=data_path, config_data=config_data)
    return DataLoader(
        dataset=dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=seed_worker,
        generator=train_generator if shuffle else general_generator,
    )


def build_data_loaders(config):
    data_loader_dict = {}

    if config.TOOLBOX_MODE == "train_and_test":
        data_loader_dict["train"] = _build_loader(
            config.TRAIN.DATA.DATASET,
            "train",
            config.TRAIN.DATA.DATA_PATH,
            config.TRAIN.DATA,
            config.TRAIN.BATCH_SIZE,
            shuffle=True,
            num_workers=16,
        )
        if config.VALID.DATA.DATASET is None and not config.TEST.USE_LAST_EPOCH:
            raise ValueError("Validation dataset not specified despite USE_LAST_EPOCH=False.")
        data_loader_dict["valid"] = None
        if config.VALID.DATA.DATASET and config.VALID.DATA.DATA_PATH and not config.TEST.USE_LAST_EPOCH:
            data_loader_dict["valid"] = _build_loader(
                config.VALID.DATA.DATASET,
                "valid",
                config.VALID.DATA.DATA_PATH,
                config.VALID.DATA,
                config.TRAIN.BATCH_SIZE,
                shuffle=False,
                num_workers=16,
            )

    if config.TOOLBOX_MODE in ("train_and_test", "only_test"):
        data_loader_dict["test"] = _build_loader(
            config.TEST.DATA.DATASET,
            "test",
            config.TEST.DATA.DATA_PATH,
            config.TEST.DATA,
            config.INFERENCE.BATCH_SIZE,
            shuffle=False,
            num_workers=4,
        )
        if config.TOOLBOX_MODE == "train_and_test" and config.TEST.USE_LAST_EPOCH:
            print("Testing uses last epoch; validation dataset is not required.\n")

    return data_loader_dict


def train_and_test(config, data_loader_dict):
    if config.MODEL.NAME != "SimpleMamba":
        raise ValueError("UG-rPPG release only supports MODEL.NAME = SimpleMamba.")
    model_trainer = trainer.SimpleMambaTrainer_260524.SimpleMambaTrainer(config, data_loader_dict)
    model_trainer.train(data_loader_dict)
    model_trainer.test(data_loader_dict)


def test(config, data_loader_dict):
    if config.MODEL.NAME != "SimpleMamba":
        raise ValueError("UG-rPPG release only supports MODEL.NAME = SimpleMamba.")
    model_trainer = trainer.SimpleMambaTrainer_260524.SimpleMambaTrainer(config, data_loader_dict)
    model_trainer.test(data_loader_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
    args = parser.parse_args()

    config = get_config(args)
    print("Configuration:")
    print(config, end="\n\n")

    data_loader_dict = build_data_loaders(config)

    if config.TOOLBOX_MODE == "train_and_test":
        train_and_test(config, data_loader_dict)
    elif config.TOOLBOX_MODE == "only_test":
        test(config, data_loader_dict)
    else:
        raise ValueError("UG-rPPG supports TOOLBOX_MODE = train_and_test or only_test.")
