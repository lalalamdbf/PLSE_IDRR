from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from prompt.pipeline_base import PromptForClassification


class PromptIDRC(nn.Module):
    def __init__(self, prompt_config):
        super(PromptIDRC, self).__init__()
        self.prompt_model = PromptForClassification(
            template=prompt_config.get_template(),
            plm=prompt_config.get_plm(),
            verbalizer=prompt_config.get_verbalizer(),
        )
    def forward(self,input_ids, attention_mask, token_type_ids, loss_ids, label):

        logits = self.prompt_model(input_ids, attention_mask, token_type_ids, loss_ids)
        loss = self.calc_loss(logits,label)

        return loss, torch.nn.functional.softmax(logits,dim=-1)
        
    def calc_loss(self,logits,label):
        
        targets = torch.argmax(label,dim=-1)
        loss_cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = loss_cross_entropy(logits, targets)

        return loss
    

