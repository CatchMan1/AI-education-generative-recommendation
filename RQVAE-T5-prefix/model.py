
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from transformers import T5ForConditionalGeneration, T5Config


class ProfessionalAdapter(nn.Module):
    """Cross-attention adapter that fuses professional hierarchy BERT embeddings
    with student hidden states to produce a single prefix token.

    Args:
        d_model (int): Hidden dimension of the T5 encoder.
        bert_dim (int): Dimension of the input BERT embeddings (default 768).
        num_heads (int): Number of attention heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model: int, bert_dim: int = 768, num_heads: int = 4, dropout: float = 0.1):
        super(ProfessionalAdapter, self).__init__()
        self.bert_proj = nn.Linear(bert_dim, d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, student_hidden: torch.Tensor, bert_vecs: torch.Tensor) -> torch.Tensor:
        """Compute one dynamic prefix token from student hidden states and BERT embeddings.

        Args:
            student_hidden (torch.Tensor): T5 input embeddings, shape (B, seq_len, d_model).
            bert_vecs (torch.Tensor): Top-5 BERT vectors, shape (B, 5, bert_dim).

        Returns:
            torch.Tensor: Prefix token of shape (B, 1, d_model).
        """
        kv = self.bert_proj(bert_vecs)  # (B, 5, d_model)
        attn_out, _ = self.cross_attn(student_hidden, kv, kv)  # Q=student, K=V=proj_bert
        x = self.norm1(student_hidden + attn_out)
        x = self.norm2(x + self.ffn(x))
        prefix = torch.mean(x, dim=1, keepdim=True)  # (B, 1, d_model)
        return prefix


class TIGER(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(TIGER, self).__init__()
        t5_config = T5Config(
            vocab_size=config['vocab_size'],
            num_layers=config['num_layers'],
            num_decoder_layers=config['num_decoder_layers'],
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            num_heads=config['num_heads'],
            d_kv=config['d_kv'],
            dropout_rate=config['dropout_rate'],
            feed_forward_proj=config['feed_forward_proj'],
            pad_token_id=config['pad_token_id'],
            eos_token_id=config['eos_token_id'],
            decoder_start_token_id=config['pad_token_id'],
        )
        self.model = T5ForConditionalGeneration(t5_config)

        bert_dim = config.get('bert_dim', 768)
        dropout_rate = config['dropout_rate']
        d_model = config['d_model']
        num_heads = config['num_heads']

        self.adapter_lvl1 = ProfessionalAdapter(d_model, bert_dim, num_heads, dropout_rate)
        self.adapter_lvl2 = ProfessionalAdapter(d_model, bert_dim, num_heads, dropout_rate)
        self.adapter_lvl3 = ProfessionalAdapter(d_model, bert_dim, num_heads, dropout_rate)
    
    @property
    def n_parameters(self):
      """Calculates the number of trainable parameters in the model.

      Returns:
          str: A string containing the number of embedding parameters,
          non-embedding parameters, and total trainable parameters.
      """
      num_params = lambda ps: sum(p.numel() for p in ps if p.requires_grad)
      total_params = num_params(self.parameters())
      emb_params = num_params(self.model.get_input_embeddings().parameters())
      return (
          f'#Embedding parameters: {emb_params}\n'
          f'#Non-embedding parameters: {total_params - emb_params}\n'
          f'#Total trainable parameters: {total_params}\n'
      )

    def _build_prefix_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        prof_lvl1: torch.Tensor,
        prof_lvl2: torch.Tensor,
        prof_lvl3: torch.Tensor,
    ):
        """Embed input_ids, generate 3 prefix tokens, and prepend them.

        Returns:
            inputs_embeds (torch.Tensor): (B, seq_len + 3, d_model)
            attention_mask (torch.Tensor): (B, seq_len + 3)
        """
        embeds = self.model.shared(input_ids)  # (B, seq_len, d_model)
        B = embeds.size(0)

        prefix1 = self.adapter_lvl1(embeds, prof_lvl1)  # (B, 1, d_model)
        prefix2 = self.adapter_lvl2(embeds, prof_lvl2)  # (B, 1, d_model)
        prefix3 = self.adapter_lvl3(embeds, prof_lvl3)  # (B, 1, d_model)

        prefix = torch.cat([prefix1, prefix2, prefix3], dim=1)  # (B, 3, d_model)
        inputs_embeds = torch.cat([prefix, embeds], dim=1)       # (B, seq_len+3, d_model)

        if attention_mask is not None:
            prefix_mask = torch.ones(B, 3, device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)  # (B, seq_len+3)

        return inputs_embeds, attention_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        prof_lvl1: Optional[torch.Tensor] = None,
        prof_lvl2: Optional[torch.Tensor] = None,
        prof_lvl3: Optional[torch.Tensor] = None,
    ):
      """Forward pass of the model. Returns the output logits and the loss value.

      Args:
          input_ids (torch.Tensor): Token IDs of shape (B, seq_len).
          attention_mask (torch.Tensor, optional): Mask of shape (B, seq_len).
          labels (torch.Tensor, optional): Target token IDs.
          prof_lvl1 (torch.Tensor, optional): Level-1 BERT embeddings (B, 5, 768).
          prof_lvl2 (torch.Tensor, optional): Level-2 BERT embeddings (B, 5, 768).
          prof_lvl3 (torch.Tensor, optional): Level-3 BERT embeddings (B, 5, 768).

      Returns:
          outputs (ModelOutput):
              The output of the model, which includes:
              - loss (torch.Tensor)
              - logits (torch.Tensor)
      """
      if prof_lvl1 is not None and prof_lvl2 is not None and prof_lvl3 is not None:
          inputs_embeds, attention_mask = self._build_prefix_inputs(
              input_ids, attention_mask, prof_lvl1, prof_lvl2, prof_lvl3
          )
          outputs = self.model(
              inputs_embeds=inputs_embeds,
              attention_mask=attention_mask,
              labels=labels,
          )
      else:
          outputs = self.model(
              input_ids=input_ids,
              attention_mask=attention_mask,
              labels=labels,
          )
      return outputs.loss, outputs.logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_beams: int = 20,
        prof_lvl1: Optional[torch.Tensor] = None,
        prof_lvl2: Optional[torch.Tensor] = None,
        prof_lvl3: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Generate recommendations using the model.

        Args:
            input_ids (torch.Tensor): Input tensor for the model.
            attention_mask (Optional[torch.Tensor]): Attention mask for the input.
            num_beams (int): Number of beams for beam search.
            prof_lvl1 (torch.Tensor, optional): Level-1 BERT embeddings (B, 5, 768).
            prof_lvl2 (torch.Tensor, optional): Level-2 BERT embeddings (B, 5, 768).
            prof_lvl3 (torch.Tensor, optional): Level-3 BERT embeddings (B, 5, 768).

        Returns:
            torch.Tensor: Generated output tensor.
        """
        if prof_lvl1 is not None and prof_lvl2 is not None and prof_lvl3 is not None:
            inputs_embeds, attention_mask = self._build_prefix_inputs(
                input_ids, attention_mask, prof_lvl1, prof_lvl2, prof_lvl3
            )
            return self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_length=5,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                **kwargs,
            )
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=5,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            **kwargs,
        )
