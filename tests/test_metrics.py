import os
import unittest
from pathlib import Path
from omegaconf import OmegaConf
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from rudalle import get_vae
from rudolph.inference.inference_dataloader import InferenceDatasetRetriever, fb_collate_fn
from rudolph.train.utils import create_dataset
from rudalle.tokenizer import get_tokenizer
from rudolph.model import get_rudolph_model, ruDolphModel
from rudolph.inference.inference_api import ruDolphApi


class TestMetricsCalculation(unittest.TestCase):
    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._
        # C:\Users\airen\Documents\Projects\RUDOLF\outputs\\modified_rudolph_1eposh_2tasks



    def estimate_vqa(self, results_path: str):
        pass

    # def _get_data(self, results_path: str) -> DataLoader:
    #     results_path = Path(results_path)
    #     if results_path.exists():
    #
    #     else:
    #         raise FileNotFoundError(f"Configuration file do not exists by path: {config_path} !")


if __name__ == '__main__':
    root_dir = Path().resolve().parent
    results_path = root_dir.parent / "outputs" / "results_2tasks" / "modified_rudolph_1eposh_2tasks"
    # vqa files
    vqa_gt_filename = results_path / "gt_test_vqa_data_p0.json"
    # captioning files
    cap_gt_filename = results_path / "gt_test_cap_data_p0.json"
    unittest.main()
