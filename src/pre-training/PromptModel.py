from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead, RobertaForMaskedLM
from torch.nn import  CrossEntropyLoss
from transformers.modeling_outputs import ModelOutput
from typing import Optional, Tuple
import torch.nn.functional as FF
import torch.nn as nn
import torch
import math

class PromptMaskedLMOutput(ModelOutput):
    mutual_loss: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    
class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):

        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = FF.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
    
class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1).repeat(1,self.h,1).unsqueeze(2)
            
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x) 
    
class Max_Discriminator(nn.Module):
    """
    Discriminator for calculating mutual information maximization
    """

    def __init__(self, hidden, initrange=None):
        super().__init__()
        self.l1 = nn.Linear(2*hidden, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 1)
        self.act = FF.relu

    def forward(self, x1, x2):
        h = torch.cat((x1, x2), dim=1)
        h = self.l1(h)
        h = self.act(h)
        h = self.l2(h)
        h = self.act(h)
        h = self.l3(h)
        return h
        
class RobertaForPrompt(RobertaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.multi_head_attentention_pooling = MultiHeadedAttention(6, config.hidden_size)
        self.max_d = Max_Discriminator(config.hidden_size)
        
        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])
        self.init_weights()

    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        conns_index=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        mutual_loss= None
        masked_lm_loss = None
         
        prediction_scores = self.lm_head(sequence_output)
        
        # Global Logical Semantics Enhancement
        if conns_index is not None:
            conn_embedding = torch.gather(sequence_output, 1, conns_index.unsqueeze(-1).expand(-1, -1, sequence_output.size(-1))).squeeze(1)
            new_mask = attention_mask.clone()
            zeros = torch.zeros((new_mask.size(0),new_mask.size(1)),dtype=torch.long,device=new_mask.device)
            new_mask.scatter_(1,conns_index,zeros)
            logic_semantic = self.multi_head_attentention_pooling(conn_embedding, sequence_output, sequence_output, new_mask).squeeze(1)
            neg_logic_semantic = torch.cat((logic_semantic[1:], logic_semantic[0].unsqueeze(0)), dim=0)
            Ej = -FF.softplus(-self.max_d(conn_embedding, logic_semantic)).mean()
            Em = FF.softplus(self.max_d(conn_embedding, neg_logic_semantic)).mean()
            mutual_loss = Em - Ej
            
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        
        return PromptMaskedLMOutput(
            mutual_loss=mutual_loss,
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )