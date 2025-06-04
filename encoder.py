import torch
from transformers import RobertaTokenizer, RobertaModel


def encode_input(input):
    """
    Encodes the input text using the provided tokenizer.
    :param text: The input text to be encoded.
    :param tokenizer: The tokenizer object.
    :return: Tuple containing input_ids and attention_mask tensors.
    """
    x = input['input_ids'].squeeze().cuda()
    y = input['attention_mask'].squeeze().cuda()
    if x.ndim == 1:
        return x.unsqueeze(0), y.unsqueeze(0)
    else:
        return x, y
