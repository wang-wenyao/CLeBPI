import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from transformers import BertLayerNorm
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
# from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
#                           BertConfig, BertForMaskedLM, BertTokenizer,
#                           GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
#                           OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
#                           RobertaConfig, RobertaModel, RobertaTokenizer,
#                           DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
from transformers.modeling_utils import PreTrainedModel


class ModelContra(PreTrainedModel):
    def __init__(self, encoder, config, tokenizer, args):
        super(ModelContra, self).__init__(config)
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        # MLP
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        # cos
        self.cos = nn.CosineSimilarity(dim=-1)
        # loss
        self.loss_func = nn.CrossEntropyLoss()
        self.args = args

    def forward(self, inputs):
        bs = inputs['input_ids'].shape[0] / 2
        outputs = self.encoder(**inputs)[1]
        outputs = self.dense(outputs)
        outputs = self.activation(outputs)
        input1 = outputs[:bs]
        input2 = outputs[bs:]

        cos_sim = self.cos(input1, input2)
        labels = torch.arange(bs).long().to(self.args.device)
        loss = self.loss_func(cos_sim, labels)

        return loss