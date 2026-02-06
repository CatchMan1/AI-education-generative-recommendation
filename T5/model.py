import torch
from transformers import T5Config, T5EncoderModel
from typing import Optional, Dict, Any
import torch.nn as nn
import torch.nn.functional as F
class TIGER(nn.Module):
    def __init__(self, params: Dict[str, Any]):
        super(TIGER, self).__init__()
        t5config = T5Config(
        num_layers=params['num_layers'],
        d_model=params['d_model'],
        d_ff=params['d_ff'],
        num_heads=params['num_heads'],
        d_kv=params['d_kv'],
        dropout_rate=params['dropout_rate'],
        feed_forward_proj=params['feed_forward_proj'],
    )
        self.model = T5EncoderModel(t5config)
        self.input_proj = nn.Linear(params['input_emb_dim'], params['d_model'])
        self.output_proj = nn.Linear(params['d_model'], params['target_emb_dim'])
        self.cosine_eps = 1e-8
        self.temperature = float(params.get('temperature', 0.07))
    
    @property
    def n_parameters(self):
      num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
      total_params = num_params(self.parameters())
      emb_params = num_params(self.model.get_input_embeddings().parameters())
      return (
          f'#Embedding parameters: {emb_params}\n'
          f'#Non-embedding parameters: {total_params - emb_params}\n'
          f'#Total trainable parameters: {total_params}\n'
      )

    def compute_contrastive_loss(self, pred_emb: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
      # In-batch contrastive loss (InfoNCE).
      pred = F.normalize(pred_emb, p=2, dim=1, eps=self.cosine_eps)
      target = F.normalize(target_emb, p=2, dim=1, eps=self.cosine_eps)

      logits = pred @ target.T
      logits = logits / self.temperature
      labels = torch.arange(logits.size(0), device=logits.device)

      loss_i2t = F.cross_entropy(logits, labels)
      loss_t2i = F.cross_entropy(logits.T, labels)
      return (loss_i2t + loss_t2i) / 2.0

    def forward(self, seq_embs: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, target_emb: Optional[torch.Tensor] = None):

      inputs_embeds = self.input_proj(seq_embs)
      outputs = self.model(
          inputs_embeds=inputs_embeds,
          attention_mask=attention_mask,
      )
      hidden = outputs.last_hidden_state
      pooled = hidden[:, 0, :]
      pred_emb = self.output_proj(pooled)
      loss = None
      if target_emb is not None:
          loss = self.compute_contrastive_loss(pred_emb, target_emb)
      pred_emb_norm = F.normalize(pred_emb, p=2, dim=1, eps=self.cosine_eps)
      return loss, pred_emb_norm
    
    def generate(self, seq_embs: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, **kwargs):
        _, pred_emb = self.forward(seq_embs=seq_embs, attention_mask=attention_mask, target_emb=None)
        return pred_emb
