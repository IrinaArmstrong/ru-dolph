import torch
from torch.utils.data import Dataset

from PIL import Image
import torchvision.transforms as T
from torch.nn.utils.rnn import pad_sequence

import youtokentome as yttm

DEFAULT_SPC_TOKENS = {
    '<LT_UNK>': 16384,
    '<RT_UNK>': 16385,

    '<LT_T2I>': 16386,
    '<LT_I2T>': 16387,
    '<LT_T2T>': 16388,
    '<RT_I2T>': 16389
}


class InferenceDatasetRetriever(Dataset):
    spc_id = -1

    def __init__(self, left_text, image_path, ids, tokenizer, model_params, labels = None):
        self.ids = ids
        self.left_text = left_text
        self.labels = labels
        self.image_path = image_path
        self.tokenizer = tokenizer
        self.model_params = model_params
        self.image_size = self.model_params.image_tokens_per_dim * 8
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(self.image_size, scale=(1., 1.), ratio=(1., 1.)),
            T.ToTensor()
        ])
        self.text_special_tokens = 1
        self.spc_tokens = DEFAULT_SPC_TOKENS
        self.decode_ignore_ids = [
            self.tokenizer.eos_id, self.tokenizer.bos_id,
            self.tokenizer.unk_id, self.tokenizer.pad_id,
            self.spc_id, *list(self.spc_tokens.values())
        ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # Only <LT_UNK> special token is available on inference stage
        left_special_token = '<LT_UNK>'
        left_text = self.left_text[idx]
        left_text = left_text.lower().strip()
        left_encoded_text = self.encode_text(left_text, text_seq_length=self.model_params.l_text_seq_length)
        left_encoded_text[torch.where(left_encoded_text == self.spc_id)] = self.spc_tokens[left_special_token]
        # as [ <BOS> <LT_UNK> <TEXT_TOKEN_#0> ... <EOS>]

        if self.image_path[idx]:
            image_path = self.image_path[idx]
            image = Image.open(image_path)
            image = self.image_transform(image)
        else:
            image = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)

        return self.ids[idx], left_encoded_text, image

    def encode_text(self, text, text_seq_length):
        """
        Encode text into format:
       [
           <BOS> <LT_SPEC_TOKEN> <TEXT_TOKEN_#0> ... <EOS>
       ]
        """
        tokens = self.tokenizer.tokenizer.encode([text], output_type=yttm.OutputType.ID)[0]
        bos = [self.tokenizer.bos_id]
        if self.text_special_tokens:
            bos.append(self.spc_id)
        tokens = bos + tokens + [self.tokenizer.eos_id]
        return self.tokenizer.prepare_tokens(tokens, text_seq_length)

    def decode_text(self, encoded, ignore_ids):
        """
        Decode tokens sequence to text string.
        """
        return self.tokenizer.tokenizer.decode(encoded.cpu().numpy().tolist(),
                                               ignore_ids=self.decode_ignore_ids)[0]


def fb_collate_fn(batch):
    """
    Reduced task number collate fn
    """
    ids, left_texts, images = [], [], []

    for i, sample in enumerate(batch):
        ids.append(sample[0])
        left_texts.append(sample[1])
        images.append(sample[2])

    ids = torch.Tensor(ids)
    left_texts = pad_sequence(left_texts, batch_first=True)
    images = torch.stack(images)

    return ids, left_texts, images
