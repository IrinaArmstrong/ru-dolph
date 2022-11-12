# -*- coding: utf-8 -*-
import more_itertools
from copy import deepcopy
from tqdm.auto import tqdm
from einops import rearrange
from collections import Counter
from typing import Tuple

import torch
import transformers
import youtokentome as yttm
import torch.nn.functional as F
import torchvision.transforms as T

DEFAULT_SPC_TOKENS = {
    '<LT_UNK>': 16384,
    '<RT_UNK>': 16385,
    '<LT_T2I>': 16386,
    '<LT_I2T>': 16387,
    '<LT_T2T>': 16388,
    '<RT_I2T>': 16389,

    '<LT_TQA>': 16390,
    '<RT_TQA>': 16391,

    '<LT_MQA>': 16392,
    '<RT_MQA>': 16393,

    '<LT_VQA>': 16394,
    '<RT_VQA>': 16395,

    '<LT_CAP>': 16396,
    '<RT_CAP>': 16397,

    '<LT_GEN>': 16398,

    '<LT_REC>': 16399,
    '<RT_REC>': 16400,
}


class ruDolphApi:
    spc_id = -1

    def __init__(self, model, tokenizer, vae, spc_tokens=None, quite=False, *, bs=24, q=0.5, txt_top_k=64,
                 img_top_k=768, txt_top_p=0.8, img_top_p=0.99, txt_temperature=0.9, img_temperature=1.0):
        self.spc_tokens = spc_tokens or deepcopy(DEFAULT_SPC_TOKENS)
        self.model = model
        self.tokenizer = tokenizer
        self.vae = vae
        ###
        self.vocab_size = self.model.get_param('vocab_size')
        self.l_text_seq_length = self.model.get_param('l_text_seq_length')
        self.r_text_seq_length = self.model.get_param('r_text_seq_length')
        self.image_seq_length = self.model.get_param('image_seq_length')
        self.image_tokens_per_dim = self.model.get_param('image_tokens_per_dim')
        self.text_special_tokens = self.model.get_param('text_special_tokens')
        self.image_special_tokens = self.model.get_param('image_special_tokens')
        self.total_seq_length = self.l_text_seq_length + self.image_seq_length + self.r_text_seq_length
        self.text_vocab_size = self.vocab_size - self.l_text_seq_length - self.text_special_tokens
        self.image_size = self.image_tokens_per_dim * 8
        self.device = self.model.get_param('device')
        self.image_transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.RandomResizedCrop(self.image_size, scale=(1., 1.), ratio=(1., 1.)),
            T.ToTensor()
        ])
        ###
        self.bs = bs
        self.q = q
        self.txt_top_k = txt_top_k
        self.txt_top_p = txt_top_p
        self.txt_temperature = txt_temperature
        self.img_top_p = img_top_p
        self.img_top_k = img_top_k
        self.img_temperature = img_temperature
        self.ignore_ids = [
            self.tokenizer.eos_id, self.tokenizer.bos_id, self.tokenizer.unk_id, self.tokenizer.pad_id,
            self.spc_id, *list(self.spc_tokens.values())
        ]
        self.quite = quite

    def generate_codebooks(self, images, image_tokens, left_text, vocab_size, top_k, top_p, temperature=1.0,
                           use_cache=True):
        """
        Generate image tokens.
        """
        torch.cuda.empty_cache()
        left_text_wo_images = []
        idxs = []
        for i in range(images.shape[0]):
            if (torch.nonzero(images[i]).shape[0] == 0):
                left_text_wo_images.append(left_text[i])
                idxs.append(i)

        if idxs:
            out = torch.stack(left_text_wo_images)
            codebooks = []
            with torch.no_grad():
                chunk_bs = out.shape[0]
                attention_mask = self.get_attention_mask(chunk_bs)
                cache = None

                iter_range = range(self.l_text_seq_length, self.l_text_seq_length + self.image_seq_length)

                if not self.quite:
                    iter_range = tqdm(iter_range)

                for idx in iter_range:
                    idx -= self.l_text_seq_length
                    logits, cache = self.model(out, attention_mask, cache=cache, use_cache=use_cache, return_loss=False)
                    logits = logits[:, -1, self.vocab_size:]
                    if self.image_special_tokens:
                        logits = logits[:, :-self.image_special_tokens]
                    logits /= temperature
                    filtered_logits = transformers.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                    probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
                    sample = torch.multinomial(probs, 1)
                    out = torch.cat((out, sample), dim=-1)
                codebooks.append(out[:, -self.image_seq_length:])
            images_codebooks = torch.cat(codebooks)

            ## Join generated images with initial images
            i = 0
            for idx in idxs:
                image_tokens[idx] = images_codebooks[i]
                i += 1

        return image_tokens, idxs

    def generate_special_tokens_embeddings(self, image_tokens: torch.Tensor, left_text: torch.Tensor,
                                           use_cache: bool = True, template: str = '',
                                           special_token: str = '<RT_UNK>') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates left and right special tokens embeddings.
        return
            left_sp_token_embeddings: tensor of size [bs, 1024] for <LT_xxx>
            right_sp_token_embeddings: tensor of size [bs, 1024] for <RT_xxx>
        """
        torch.cuda.empty_cache()
        chunk_bs = left_text.shape[0]

        template = template.lower().strip()
        template_encoded = self.encode_text(template, text_seq_length=self.l_text_seq_length)

        template_size = (template_encoded != 0).sum() - 1  # eos

        template_encoded = template_encoded[:template_size]
        template_encoded[torch.where(template_encoded == self.spc_id)] = self.spc_tokens[special_token]
        # template_encoded as: [bos, <RT_UNK>] == [2, 16385], of size: [2,]
        # print('Decoded sequence', self.decode_text(template_encoded))

        with torch.no_grad():
            attention_mask = self.get_attention_mask(chunk_bs)

            out = torch.cat((
                left_text.to(self.device),  # of size: [bs, 64]
                image_tokens,  # of size: [bs, 256]
                template_encoded.repeat(chunk_bs, 1).to(self.device),  # of size: [bs, 2]
            ), dim=1)
            # out of size: [2, 322]

            iter_range = range(
                self.l_text_seq_length + self.image_seq_length + template_size,
                self.l_text_seq_length + self.image_seq_length + self.r_text_seq_length
            )  # iter_range: (322, 384) - for tokens generation

            cache = None
            outputs = self.model(out, attention_mask, cache=cache, use_cache=use_cache,
                                 return_loss=False, return_hidden_states=True)
            # outputs - 3 tensors:
            # logits of size: [bs, input_seq_len==322, vocab_size==25408]
            # cache
            # all hidden states of size: [24] & [bs, input_seq_len==322, 1024]
            last_hidden_states = outputs[-1][-1].detach().cpu()
            lt_sp_token_embeddings = last_hidden_states[:, 1, :]
            rt_sp_token_embeddings = last_hidden_states[:, -1, :]

            return lt_sp_token_embeddings, rt_sp_token_embeddings

    def generate_tokens(self, image_tokens, left_text, vocab_size, top_k, top_p, temperature=1.0, use_cache=True,
                        template='', allowed_token_ids=None, special_token='<RT_UNK>'):
        """
            Generate right text tokens.
        """
        torch.cuda.empty_cache()
        generated_tokens = []
        chunk_bs = left_text.shape[0]

        template = template.lower().strip()
        template_encoded = self.encode_text(template, text_seq_length=self.l_text_seq_length)

        template_size = (template_encoded != 0).sum() - 1  # eos

        template_encoded = template_encoded[:template_size]
        template_encoded[torch.where(template_encoded == self.spc_id)] = self.spc_tokens[special_token]

        with torch.no_grad():
            attention_mask = self.get_attention_mask(chunk_bs)

            out = torch.cat((
                left_text.to(self.device),
                image_tokens,
                template_encoded.repeat(chunk_bs, 1).to(self.device),
            ), dim=1)

            cache = None
            iter_range = range(
                self.l_text_seq_length + self.image_seq_length + template_size,
                self.l_text_seq_length + self.image_seq_length + self.r_text_seq_length
            )

            if not self.quite:
                iter_range = tqdm(iter_range)

            for _ in iter_range:
                # print(cache, use_cache)
                logits, cache = self.model(out, attention_mask, cache=cache, use_cache=use_cache, return_loss=False)

                logits = logits[:, -1, :self.vocab_size]

                if allowed_token_ids:
                    logits = logits[:, allowed_token_ids]

                logits /= temperature
                filtered_logits = transformers.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
                sample = torch.multinomial(probs, 1)

                if allowed_token_ids:
                    sample = torch.tensor(allowed_token_ids).to(self.device)[sample]

                indexes = torch.where(sample >= self.vocab_size - self.l_text_seq_length)
                sample[indexes] = 3
                out = torch.cat((out, sample), dim=-1)

            generated_tokens.append(out[:, -self.r_text_seq_length:])

        tokens = torch.cat(generated_tokens)[:, 1:]
        # tokens = get_out_mask(tokens, self.device)

        texts = []

        for i in range(tokens.shape[0]):
            end = torch.where(tokens[i] == 3)[0]
            if (len(end) > 0):
                end = end[0]
            else:
                end = tokens[i].shape[0]

            text = self.decode_text(tokens[i][:end + 1]).strip()  # end+1
            if text:
                texts.append(text)
            else:
                texts.append('пустая строка')
        '''
        for tokens in generated_tokens:
            end = torch.where(tokens == 3)[0]
            if (len(end) > 0):
                end = end[0]
            else:
                end = tokens.shape[0]
            
            #end = torch.where(tokens == 3)[0][0] or tokens.shape[0]
            text = self.decode_text(tokens[:end+1]).strip()#end+1
            if text:
                texts.append(text)
        '''
        # for i in range(tokens.shape[0]):
        #    texts.append(self.decode_text(tokens[i]).strip())
        return texts

    def generate_tokens_left(self, image_tokens, left_text, vocab_size, top_k, top_p, temperature=1.0, use_cache=True,
                             template='', allowed_token_ids=None, special_token='<LT_TQA>'):
        '''
            Generate right text tokens.
        '''
        torch.cuda.empty_cache()
        generated_tokens = []
        chunk_bs = left_text.shape[0]

        template = template.lower().strip()
        template_encoded = self.encode_text(template, text_seq_length=self.l_text_seq_length, add_special_token=False)

        template_size = (template_encoded != 0).sum() - 1  # eos

        template_encoded = template_encoded[:template_size]
        template_encoded[torch.where(template_encoded == self.spc_id)] = self.spc_tokens[special_token]

        # print(left_text.shape)

        left_text[left_text == 3] = 0  # eos
        left_text_size = max((left_text != 0).sum(1))
        left_text = left_text[:, :left_text_size]

        if not self.quite:
            print('--> template_size:', template_size.item() + left_text_size)

        with torch.no_grad():
            attention_mask = self.get_attention_mask(chunk_bs)

            out = torch.cat((
                left_text.to(self.device),
                template_encoded.repeat(chunk_bs, 1).to(self.device)
            ), dim=1)

            # print('Out shape', out.shape)

            cache = None
            iter_range = range(
                left_text_size + template_size,
                self.l_text_seq_length
            )
            # print(left_text_size + template_size, self.l_text_seq_length)

            if not self.quite:
                iter_range = tqdm(iter_range)

            for _ in iter_range:
                # print(cache, use_cache)
                logits, cache = self.model(out, attention_mask, cache=cache, use_cache=use_cache, return_loss=False)

                logits = logits[:, -1, :self.vocab_size]

                if allowed_token_ids:
                    logits = logits[:, allowed_token_ids]

                logits /= temperature
                filtered_logits = transformers.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
                sample = torch.multinomial(probs, 1)

                if allowed_token_ids:
                    sample = torch.tensor(allowed_token_ids).to(self.device)[sample]

                indexes = torch.where(sample > self.text_vocab_size)
                sample[indexes] = self.tokenizer.eos_id
                out = torch.cat((out, sample), dim=-1)

            generated_tokens.append(out[:, :self.l_text_seq_length])

        tokens = torch.cat(generated_tokens)[:, 1:]
        # tokens = get_out_mask(tokens, self.device)

        texts = []
        for i in range(tokens.shape[0]):
            texts.append(self.decode_text(tokens[i]).strip())
        return texts

    def generate_texts(
            self, template='',
            top_k=None, top_p=None, texts_num=48,
            early_stop=None,
            temperature=None, bs=None, seed=None, use_cache=True, special_token='<LT_T2T>',
            allowed_token_ids=None,
    ):
        torch.cuda.empty_cache()
        bs = bs or self.bs
        top_k = top_k or self.txt_top_k
        top_p = top_p or self.txt_top_p
        temperature = temperature or self.txt_temperature

        early_stop = early_stop or self.l_text_seq_length

        template = template.lower().strip()
        template_encoded = self.encode_text(template, text_seq_length=self.l_text_seq_length, add_special_token=False)
        template_size = (template_encoded != 0).sum() - 1  # eos
        if not self.quite:
            print('--> template_size:', template_size.item())
        template_encoded = template_encoded[:template_size]
        template_encoded[torch.where(template_encoded == self.spc_id)] = self.spc_tokens[special_token]

        generated_tokens = []
        for chunk in more_itertools.chunked(range(texts_num), bs):
            chunk_bs = len(chunk)
            with torch.no_grad():
                attention_mask = self.get_attention_mask(chunk_bs)
                out = template_encoded.repeat(chunk_bs, 1).to(self.device)
                cache = None
                iter_range = range(template_size, min(early_stop, self.l_text_seq_length))
                if not self.quite:
                    iter_range = tqdm(iter_range)
                for _ in iter_range:
                    logits, cache = self.model(out, attention_mask, cache=cache, use_cache=use_cache, return_loss=False)
                    logits = logits[:, -1, :self.vocab_size]
                    if allowed_token_ids:
                        logits = logits[:, allowed_token_ids]
                    logits /= temperature
                    filtered_logits = transformers.top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                    probs = torch.nn.functional.softmax(filtered_logits, dim=-1)
                    sample = torch.multinomial(probs, 1)
                    if allowed_token_ids:
                        sample = torch.tensor(allowed_token_ids).to(self.device)[sample]
                    indexes = torch.where(sample > self.text_vocab_size)
                    sample[indexes] = self.tokenizer.eos_id
                    out = torch.cat((out, sample), dim=-1)

                generated_tokens.append(out[:, :self.l_text_seq_length])

        generated_tokens = torch.cat(generated_tokens)

        texts = Counter()
        for tokens in generated_tokens:
            end = torch.where(tokens == self.tokenizer.eos_id)[0]
            if end.shape[0]:
                end = min(end)
            else:
                end = tokens.shape[0]

            text = self.decode_text(tokens[:end + 1]).strip()
            if text:
                texts.update([text])

        texts = list(texts.items())

        ppl_txt = []
        for chunk in more_itertools.chunked(texts, bs):
            chunk_bs = len(chunk)
            with torch.no_grad():
                chunk_encoded = []
                for text, _ in chunk:
                    text = text.lower().strip()
                    encoded = self.encode_text(text, text_seq_length=self.l_text_seq_length, add_special_token=False)
                    chunk_encoded.append(encoded)

                chunk_encoded = torch.stack(chunk_encoded)
                chunk_encoded[torch.where(chunk_encoded == self.spc_id)] = self.spc_tokens[special_token]

                attention_mask = self.get_attention_mask(chunk_bs)
                input_ids = chunk_encoded.to(self.device)

                logits, _ = self.model(input_ids, attention_mask, cache=None, use_cache=False, return_loss=False)
                logits = rearrange(logits, 'b n c -> b c n')

                l_text_logits = logits[:, :self.vocab_size - self.l_text_seq_length,
                                :self.l_text_seq_length - 1].contiguous().float()
                input_ids = input_ids.contiguous().long()

                ppl_txt.append(
                    self.ce_to_ppl(F.cross_entropy(
                        l_text_logits,
                        input_ids[:, 1:self.l_text_seq_length],
                        ignore_index=0,
                        reduction='none',
                    ))
                )

        ppl_txt = torch.cat(ppl_txt)

        result = []
        for idx in ppl_txt.argsort():
            idx = idx.item()
            text, count = texts[idx]
            result.append({
                'text': text,
                'ppl_txt': round(ppl_txt[idx].item(), 2),
                'count': count,
            })

        return result

    @staticmethod
    def ce_to_ppl(ce):
        indexes = torch.where(ce)
        ce[indexes] = torch.exp(ce[indexes])
        ppl = ce.sum(1) / torch.unique(indexes[0], return_counts=True)[1]
        return ppl

    def get_attention_mask(self, bs):
        return torch.tril(torch.ones((bs, 1, self.total_seq_length, self.total_seq_length), device=self.device))

    def encode_text(self, text, text_seq_length, add_special_token=True):
        """
        Encode text into format:
           [
               <BOS> <LT_SPEC_TOKEN> <TEXT_TOKEN_#0> ... <EOS>
           ]
        """
        tokens = self.tokenizer.tokenizer.encode([text], output_type=yttm.OutputType.ID)[0]
        bos = [self.tokenizer.bos_id]
        if add_special_token:
            bos.append(self.spc_id)
        tokens = bos + tokens + [self.tokenizer.eos_id]
        return self.tokenizer.prepare_tokens(tokens, text_seq_length)

    def decode_text(self, encoded):
        """
        Decode tokens sequence to text string.
        """
        return self.tokenizer.tokenizer.decode(encoded.cpu().numpy().tolist(), ignore_ids=self.ignore_ids)[0]
