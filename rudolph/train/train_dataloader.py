import random

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from PIL import Image
import torchvision.transforms as T
import youtokentome as yttm

DEFAULT_SPC_TOKENS = {
    '<LT_UNK>': 16384,
    '<RT_UNK>': 16385,
    '<LT_T2I>': 16386,
    '<LT_I2T>': 16387,
    '<LT_T2T>': 16388,
    '<RT_I2T>': 16389,

    '<LT_VQA>': 16394,
    '<RT_VQA>': 16395,

    '<LT_CAP>': 16396,
    '<RT_CAP>': 16397
}

tokens_mapping = {
    'captioning': ['<LT_CAP>', '<RT_CAP>'],
    'vqa': ['<LT_VQA>', '<RT_VQA>']
}


class TrainDatasetRetriever(Dataset):
    spc_id = -1

    def __init__(self, task_ids, left_texts, image_paths, right_texts, stage, tokenizer, model_params):
        self.task_ids = task_ids
        self.left_texts = left_texts
        self.image_paths = image_paths
        self.right_texts = right_texts
        self.stage = stage
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
        self.tokens_mapping = tokens_mapping
        self.decode_ignore_ids = [
            self.tokenizer.eos_id, self.tokenizer.bos_id,
            self.tokenizer.unk_id, self.tokenizer.pad_id,
            self.spc_id, *list(self.spc_tokens.values())
        ]

    def __len__(self):
        return len(self.task_ids)

    def __getitem__(self, idx):
        # Adds task-specific special token id
        task_id = self.task_ids[idx]

        left_text = self.left_texts[idx]
        left_text = left_text.lower().strip()
        left_encoded_text = self.encode_text(left_text, text_seq_length=self.model_params.l_text_seq_length)
        # randomly mask 25% of left text special tokens with '<LT_UNK>'
        # left_special_token = random.choices([self.tokens_mapping[task_id][0], '<LT_UNK>'], weights=[0.75, 0.25])[0]
        left_special_token = self.tokens_mapping[task_id][0]
        left_encoded_text[torch.where(left_encoded_text == self.spc_id)] = self.spc_tokens[left_special_token]

        if self.image_paths[idx]:
            image_path = self.image_paths[idx]
            image = Image.open(image_path)
            image = self.image_transform(image)
        else:
            image = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)

        if self.right_texts[idx]:
            right_text = self.right_texts[idx]
            right_text = right_text.lower().strip()
            right_encoded_text = self.encode_text(right_text, text_seq_length=self.model_params.r_text_seq_length)
            # randomly mask 25% of right text special tokens with '<LT_UNK>'
            # right_special_token = random.choices([self.tokens_mapping[task_id][1], '<RT_UNK>'], weights=[0.75, 0.25])[0]
            right_special_token = self.tokens_mapping[task_id][1]
            right_encoded_text[torch.where(right_encoded_text == self.spc_id)] = self.spc_tokens[right_special_token]
        else:
            right_encoded_text = torch.zeros(self.model_params.r_text_seq_length, dtype=torch.int32)

        return {
            'task_id': task_id,
            'left_text': left_encoded_text,
            'image': image,
            'right_text': right_encoded_text
        }

    def get_task_labels(self):
        return list(self.task_ids)

    def encode_text(self, text, text_seq_length, add_special=True):
        """
        Encode text into format:
        [
            <BOS> <LT_SPEC_TOKEN> <TEXT_TOKEN_#0> ... <EOS>
        ]
        """
        tokens = self.tokenizer.tokenizer.encode([text], output_type=yttm.OutputType.ID)[0]
        bos = [self.tokenizer.bos_id]
        if add_special:
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
    left_text_c, image_c, right_text_c = [], [], []
    left_text_vqa, image_vqa, right_text_vqa = [], [], []

    for i, sample in enumerate(batch):
        if sample['task_id'] == 'captioning':
            left_text_c.append(sample['left_text'])
            image_c.append(sample['image'])
            right_text_c.append(sample['right_text'])
        elif sample['task_id'] == 'vqa':
            left_text_vqa.append(sample['left_text'])
            image_vqa.append(sample['image'])
            right_text_vqa.append(sample['right_text'])

    if left_text_c:
        left_text_c = pad_sequence(left_text_c, batch_first=True)
        image_c = torch.stack(image_c)
        right_text_c = pad_sequence(right_text_c, batch_first=True)

    if left_text_vqa:
        left_text_vqa = pad_sequence(left_text_vqa, batch_first=True)
        image_vqa = torch.stack(image_vqa)
        right_text_vqa = pad_sequence(right_text_vqa, batch_first=True)

    return (left_text_c, image_c, right_text_c), \
           (left_text_vqa, image_vqa, right_text_vqa)
