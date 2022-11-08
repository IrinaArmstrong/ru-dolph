import os
from pathlib import Path
from omegaconf import OmegaConf
import pandas as pd
from tqdm import tqdm
from typing import Optional

import torch
from torch.utils.data import DataLoader
from rudalle import get_vae
from rudolph.inference.inference_dataloader import InferenceDatasetRetriever, fb_collate_fn
from rudolph.train.utils import create_dataset
from rudalle.tokenizer import get_tokenizer
from rudolph.model import get_rudolph_model
from rudolph.inference.inference_api import ruDolphApi


class EmbeddingsGenerator:

    def __init__(self, config_path: str, model: Optional = None,
                 tokenizer: Optional = None, vae: Optional = None):
        self._root_dir = Path().resolve().parent
        if Path(config_path).exists():
            self.config = OmegaConf.load(config_path)
            print(OmegaConf.to_yaml(self.config))
        else:
            raise FileNotFoundError(f"Configuration file do not exists by path: {config_path} !")
        self.device = self.config['model'].rudolph.device
        self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer()
        self.vae = vae if vae is not None else get_vae(dwt=False).to(self.device)
        self.model = model if model is not None else get_rudolph_model('350M', fp16=False, device=self.device)
        self.vocab_size = self.model.get_param('vocab_size')
        self.api = ruDolphApi(self.model, self.tokenizer, self.vae, bs=12)

    def _get_dataloader(self, task_name: str, df: pd.DataFrame = None) -> DataLoader:
        task_config = self.config['data'][task_name]
        if df is None:
            df = create_dataset('captioning',
                                dataset_path=task_config['dataset_path'],
                                train_input=task_config['train_input'],
                                train_output=task_config['train_output'],
                                val_input=task_config['val_input'],
                                val_output=task_config['val_output'])

            df = pd.DataFrame(df)
        else:
            print(f"Loaded pre-defined dataset...")
        print(f"Image captioning data: {df.shape}\n")

        # Datasets creation (requires tokenizer definition)
        ds = InferenceDatasetRetriever(
            ids=df.index.values,
            left_text=df['left_text'].values,
            image_path=df['image_path'].values,
            labels=df['task_id'].values,
            tokenizer=self.tokenizer,
            model_params=self.config['model'].params
        )
        print(f"Dataset size: {len(ds)}")

        # Dataloader
        loader = DataLoader(
            ds,
            batch_size=self.config.bs,
            pin_memory=False,
            drop_last=False,
            collate_fn=fb_collate_fn
        )
        print(f"Dataloader size: {len(loader)} with batch size: {self.config.bs}")
        return loader

    def generate_sp_tokens_embeddings(self, task_name: str = 'captioning', df: pd.DataFrame = None):
        loader = self._get_dataloader(task_name=task_name, df=df)

        vocab = self.tokenizer.tokenizer.vocab()
        allowed_token_ids = []
        for i, token in enumerate(vocab):
            allowed_token_ids.append(i)

        sp_tokens_embeddings = []
        gt_labels = loader.dataset.labels
        for batch in tqdm(loader):
            ids_question, left_text, images = batch
            left_text = left_text.to(self.device)
            images = images.to(self.device)
            image_tokens = self.vae.get_codebook_indices(images)

            # Generate right texts
            l_lhs, r_lhs = self.api.generate_special_tokens_embeddings(image_tokens, left_text, self.vocab_size,
                                                                       template='', special_token='<RT_UNK>')

            # Save predictions to dict
            for id, l_e, r_e in zip(ids_question, l_lhs, r_lhs):
                sp_tokens_embeddings.append({
                    "id": id.item(),
                    "left_sp_token_embedding": l_e.numpy(),
                    "right_sp_token_embedding": r_e.numpy(),
                    "gt_task_label": gt_labels[int(id.item())]
                })

        return sp_tokens_embeddings
