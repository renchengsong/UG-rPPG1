"""Entry point for MC Dropout baseline training and testing."""

import argparse
import random

import numpy as np
import torch

from config import get_config
from main import build_data_loaders, add_args
from neural_methods import trainer

RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train_and_test(config, data_loader_dict):
    if config.MODEL.NAME != "SimpleMamba":
        raise ValueError("MC baseline expects MODEL.NAME = SimpleMamba.")
    model_trainer = trainer.SimpleMambaTrainer_MC.SimpleMambaTrainer(config, data_loader_dict)
    model_trainer.train(data_loader_dict)
    model_trainer.test(data_loader_dict)


def test(config, data_loader_dict):
    if config.MODEL.NAME != "SimpleMamba":
        raise ValueError("MC baseline expects MODEL.NAME = SimpleMamba.")
    model_trainer = trainer.SimpleMambaTrainer_MC.SimpleMambaTrainer(config, data_loader_dict)
    model_trainer.test(data_loader_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UG-rPPG MC Dropout baseline")
    parser = add_args(parser)
    parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
    from dataset import data_loader
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
        raise ValueError("UG-rPPG MC baseline supports TOOLBOX_MODE = train_and_test or only_test.")
