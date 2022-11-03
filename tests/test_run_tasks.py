import unittest
from pathlib import Path
import requests
from PIL import Image

import torch
from rudalle import get_vae
from rudalle.utils import seed_everything
from rudalle.tokenizer import get_tokenizer
from rudolph.model import get_rudolph_model, ruDolphModel
from rudolph.api import ruDolphApi


class TestRunTasks(unittest.TestCase):

    def __init__(self, method_name="runTest"):
        super().__init__(method_name)
        self._current_dir = Path().resolve()
        self.device = 'cpu'
        self.tokenizer = get_tokenizer()
        self.vae = get_vae(dwt=False).to(self.device)
        self.model = get_rudolph_model('350M', fp16=False, device=self.device)
        self.api = ruDolphApi(self.model, self.tokenizer, self.vae, bs=48)

    def test_image_captioning(self):
        img_by_url = 'https://raw.githubusercontent.com/sberbank-ai/ru-dolph/master/pics/pipelines/captioning_dog.png'
        pil_img = Image.open(requests.get(img_by_url, stream=True).raw).resize((192, 192))

        result = self.api.image_captioning(
            pil_img, early_stop=32,
            generations_num=48,
            captions_num=1,
            seed=42,
            l_special_token='<LT_UNK>',
            r_special_token='<RT_UNK>'
        )

        for sample in result:
            print(f'- {sample["text"]}')





if __name__ == '__main__':
    unittest.main()
