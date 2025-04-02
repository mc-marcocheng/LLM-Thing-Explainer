import gradio as gr
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          LogitsProcessorList, TextIteratorStreamer)


class LLMLoader:
    def __init__(self):
        self.model_id = None
        self.tokenizer = None
        self.model = None

    def load_model(self, model_id):
        if self.model_id == model_id:
            return
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
