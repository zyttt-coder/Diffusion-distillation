import torch
import torch.nn as nn
import numpy as np

from functools import partial
from transformers import CLIPTokenizer

def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(string, truncation=True, max_length=77, return_length=True,
                               return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    assert torch.count_nonzero(tokens - 49407) == 2, f"String '{string}' maps to more than a single token. Please use another string"

    return tokens[0, 1]

class EmbModel(nn.Module):
    def __init__(
            self,
            tokenizer,
            device,
            num_classes=10,
            ipc=1,
            placeholder="*",
            **kwargs
    ):
        super().__init__()

        self.ipc = ipc
        self.num_classes = num_classes
        self.tokenizer = tokenizer
        self.placeholder = placeholder

        self.string_to_token_dict = {}
        self.string_to_param_dict = {}

        if isinstance(self.tokenizer,CLIPTokenizer):
            # using Stable Diffusion's CLIP encoder
            token_dim = 768

            get_token_for_string = partial(get_clip_token_for_string, self.tokenizer)
            placeholder_token = get_token_for_string(placeholder)
            token_params = torch.randn(self.num_classes*self.ipc, token_dim).to(device).requires_grad_()
            
        self.string_to_token_dict[placeholder] = placeholder_token
        self.string_to_param_dict[placeholder] = token_params

    def forward(
            self,
            tokenized_text,
            embedded_text,
    ):
        device = tokenized_text.device

        for c in range(self.num_classes):
            for i in range(self.ipc):
                placeholder_embedding = self.string_to_param_dict[self.placeholder][c*self.ipc + i]
                embedded_text[c*self.ipc + i][torch.argwhere(tokenized_text == self.string_to_token_dict[self.placeholder].to(device)).squeeze()] = placeholder_embedding

    def save(self, ckpt_path):
        torch.save({"string_to_token": self.string_to_token_dict,
                    "string_to_param": self.string_to_param_dict}, ckpt_path)

    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')

        self.string_to_token_dict = ckpt["string_to_token"]
        self.string_to_param_dict = ckpt["string_to_param"]

    def embedding_parameters(self):
        return self.string_to_param_dict[self.placeholder]
    

    