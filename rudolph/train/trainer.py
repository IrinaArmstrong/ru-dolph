import torch
import pytorch_lightning as pl
import bitsandbytes as bnb

from rudolph.model.utils import get_attention_mask


class RudolphLightning(pl.LightningModule):

    def __init__(self, vae, model, n_training_steps, task_weights, model_params, model_freeze, scheduler_conf, bs):
        super().__init__()
        self.save_hyperparameters()
        self.vae = vae
        self.model = freeze(model=model, **model_freeze)
        self.n_training_steps = n_training_steps
        self.task_weights = task_weights
        self.model_params = model_params
        self.scheduler_conf = scheduler_conf
        self.bs = bs

    def forward(self, input_ids, attention_mask):
        logits, has_cache = self.model(input_ids, attention_mask, return_loss=False)
        return logits

    def get_loss(self, bs, left_text, image, right_text, task):
        attention_mask = get_attention_mask(bs, self.model_params.l_text_seq_length,
                                            self.model_params.image_tokens_per_dim, self.model_params.r_text_seq_length,
                                            left_text.device)
        if image is None:
            image_seq_length = self.model_params.image_tokens_per_dim ** 2
            image_input_ids = torch.zeros((bs, image_seq_length), dtype=torch.int32).to(left_text.device)
        else:
            image_input_ids = self.vae.get_codebook_indices(image)
        if right_text is None:
            input_ids = torch.cat((left_text, image_input_ids), dim=1)
        else:
            input_ids = torch.cat((left_text, image_input_ids, right_text), dim=1)
        loss, loss_values = self.model.forward(input_ids, attention_mask, lt_loss_weight=task.lt_loss_weight,
                                               img_loss_weight=task.img_loss_weight, rt_loss_weight=task.rt_loss_weight,
                                               sp_loss_weight=task.sp_loss_weight,
                                               return_loss=True)
        return loss

    def model_step(self, batch, stage):
        (left_text_c, image_c, right_text_c), (left_text_vqa, image_vqa, right_text_vqa) = batch

        losses = []

        ## captioning
        if len(left_text_c) > 0:
            bs_c = left_text_c.shape[0]
            loss_с = self.get_loss(bs_c, left_text_c, image_c, right_text_c, self.task_weights.capt)
            self.log(f"{stage}_loss_с", loss_с, prog_bar=True, logger=True, batch_size=bs_c)
            losses.append(self.task_weights.capt_loss_weight * loss_с)

        ## vqa
        if len(left_text_vqa) > 0:
            bs_vqa = left_text_vqa.shape[0]
            loss_vqa = self.get_loss(bs_vqa, left_text_vqa, image_vqa, right_text_vqa, self.task_weights.vqa)
            self.log(f"{stage}_loss_vqa", loss_vqa, prog_bar=True, logger=True, batch_size=bs_vqa)
            losses.append(self.task_weights.vqa_loss_weight * loss_vqa)

        ## join loss
        loss = sum(losses)
        self.log(f"{stage}_loss", loss, prog_bar=True, logger=True, batch_size=self.bs)
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        return self.model_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.model_step(batch, 'valid')

    def configure_optimizers(self):
        optimizer = bnb.optim.Adam8bit(self.model.parameters(), lr=self.scheduler_conf.max_lr)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.scheduler_conf.max_lr, final_div_factor=500,
            total_steps=self.n_training_steps
        )

        scheduler = {"scheduler": lr_scheduler, "interval": "step"}
        return [optimizer], [scheduler]


def freeze(
        model,
        freeze_emb=False,
        freeze_ln=False,
        freeze_attn=True,
        freeze_ff=True,
        freeze_other=False,
):
    for name, p in model.module.named_parameters():
        name = name.lower()
        if 'ln' in name or 'norm' in name:
            p.requires_grad = not freeze_ln
        elif 'embeddings' in name:
            p.requires_grad = not freeze_emb
        elif 'mlp' in name:
            p.requires_grad = not freeze_ff
        elif 'attn' in name:
            p.requires_grad = not freeze_attn
        else:
            p.requires_grad = not freeze_other
    return model
