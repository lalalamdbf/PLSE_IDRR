from typing import List, Optional
from transformers.modeling_utils import PreTrainedModel
from .utils import TokenizerWrapper
from transformers.tokenization_utils import PreTrainedTokenizer
from .mlm import MLMTokenizerWrapper
from transformers import BertConfig, BertTokenizer, BertModel, BertForMaskedLM, \
                         RobertaConfig, RobertaTokenizer, RobertaModel, RobertaForMaskedLM
from collections import namedtuple



ModelClass = namedtuple("ModelClass", ('config', 'tokenizer', 'model','wrapper'))


def load_plm(model_path, specials_to_add = None):
    r"""A plm loader using a global config.
    It will load the model, tokenizer, and config simulatenously.

    Args:
        config (:obj:`CfgNode`): The global config from the CfgNode.

    Returns:
        :obj:`PreTrainedModel`: The pretrained model.
        :obj:`tokenizer`: The pretrained tokenizer.
        :obj:`model_config`: The config of the pretrained model.
        :obj:`wrapper`: The wrapper class of this plm.
    """
    model_config = RobertaConfig.from_pretrained(model_path)

    model = RobertaForMaskedLM.from_pretrained(model_path, config=model_config)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    wrapper = MLMTokenizerWrapper


    model, tokenizer = add_special_tokens(model, tokenizer, specials_to_add=specials_to_add)
    
    return model, tokenizer, model_config, wrapper

def add_special_tokens(model: PreTrainedModel,
                       tokenizer: PreTrainedTokenizer,
                       specials_to_add: Optional[List[str]] = None):
    r"""add the special_tokens to tokenizer if the special token
    is not in the tokenizer.

    Args:
        model (:obj:`PreTrainedModel`): The pretrained model to resize embedding
                after adding special tokens.
        tokenizer (:obj:`PreTrainedTokenizer`): The pretrained tokenizer to add special tokens.
        specials_to_add: (:obj:`List[str]`, optional): The special tokens to be added. Defaults to pad token.

    Returns:
        The resized model, The tokenizer with the added special tokens.

    """
    if specials_to_add is None:
        return model, tokenizer
    for token in specials_to_add:
        if "pad" in token.lower():
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': token})
                model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer



