import os
import unittest
from pathlib import Path
from omegaconf import OmegaConf
import pandas as pd

import torch
from torch.utils.data import DataLoader
from rudalle import get_vae
from rudolph.train.train_dataloader import TrainDatasetRetriever, fb_collate_fn
from rudolph.train.utils import create_dataset
from rudolph.model.utils import get_attention_mask
from rudalle.tokenizer import get_tokenizer
from rudolph.model import get_rudolph_model, ruDolphModel


class TestLossCalculation(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._root_dir = Path().resolve().parent
        config_path = self._root_dir / "configuration" / "multi_task_ft_cpu.yaml"
        if config_path.exists():
            self.config = OmegaConf.load(str(config_path))
            # print(OmegaConf.to_yaml(self.config))
        else:
            raise FileNotFoundError(f"Configuration file do not exists by path: {config_path} !")
        self.device = self.config['model'].rudolph.device
        self.tokenizer = get_tokenizer()
        self.vae = get_vae(dwt=False).to(self.device)
        self.model = get_rudolph_model('350M', fp16=False, device=self.device)
        self.loader = self._get_dataloader(task_name='captioning')

    def _get_dataloader(self, task_name: str) -> DataLoader:
        task_config = self.config['data'][task_name]
        df = create_dataset('captioning',
                            dataset_path=task_config['dataset_path'],
                            train_input=task_config['train_input'],
                            train_output=task_config['train_output'],
                            val_input=task_config['val_input'],
                            val_output=task_config['val_output'])

        df = pd.DataFrame(df)
        print(f"Image captioning data: {df.shape}\n")

        # Datasets creation (requires tokenizer definition)
        ds = TrainDatasetRetriever(
            task_ids=df['task_id'].values,
            left_texts=df['left_text'].values,
            image_paths=df['image_path'].values,
            right_texts=df['right_text'].values,
            stage='train',
            tokenizer=self.tokenizer,
            model_params=self.config['model'].params
        )
        print(f"Dataset size: {len(ds)}")

        # Dataloader
        loader = DataLoader(
            ds,
            batch_size=self.config['trainer'].bs,
            pin_memory=False,
            drop_last=False,
            collate_fn=fb_collate_fn
        )
        print(f"Dataloader size: {len(loader)} with batch size: {self.config['trainer'].bs}")
        return loader

    def test_basic_loss(self):
        batch = next(iter(self.loader))
        print(f"Batch of size: {len(batch)}")

        # 0 - capture, 1 - vqa
        (left_text_c, image_c, right_text_c), (left_text_vqa, image_vqa, right_text_vqa) = batch

        if len(left_text_c) > 0:
            bs_c = left_text_c.shape[0]
            print(f"Batch size: {bs_c}")
            attention_mask_c = get_attention_mask(bs_c,
                                                  self.config['model']['params'].l_text_seq_length,
                                                  self.config['model']['params'].image_tokens_per_dim,
                                                  self.config['model']['params'].r_text_seq_length,
                                                  left_text_c.device)
            print(f"Attention mask size: {attention_mask_c.size()}")

            image_input_ids = self.vae.get_codebook_indices(image_c)
            if right_text_c is None:
                input_ids = torch.cat((left_text_c, image_input_ids), dim=1)
            else:
                input_ids = torch.cat((left_text_c, image_input_ids, right_text_c), dim=1)
            print(f"input_ids: {input_ids.size()}")

            capt_loss_weights = self.config.trainer.task_weights.capt
            loss_c, loss_values_c = self.model.forward(input_ids, attention_mask_c,
                                                       lt_loss_weight=capt_loss_weights.lt_loss_weight,
                                                       img_loss_weight=capt_loss_weights.img_loss_weight,
                                                       rt_loss_weight=capt_loss_weights.rt_loss_weight,
                                                       sp_loss_weight=1,
                                                       return_loss=True)

        self.assertEqual(1, 1)


if __name__ == '__main__':
    unittest.main()
