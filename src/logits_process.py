import torch
from transformers.generation.logits_process import (
    LOGITS_PROCESSOR_INPUTS_DOCSTRING, LogitsProcessor)
from transformers.utils import add_start_docstrings

from .state_machine import TokenStateMachine


class StateMachineLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that integrates with a token state machine to filter
    logits based on the valid next tokens. It ensures that the next token
    selected by the model is valid according to the state machine.

    Args:
        list_of_token_lists (`list[list[int]]`):
            A list of token lists, where each sublist defines a set of valid token transitions.

    Examples:

    ```python
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
    >>> from logits_process import StateMachineLogitsProcessor

    >>> tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-560m")
    >>> model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-560m")

    >>> processor = StateMachineLogitsProcessor(list_of_token_lists=[[0, 1], [2, 3, 4], [5]])

    >>> inputs = tokenizer("A number:", return_tensors="pt")
    >>> logits = model(**inputs).logits
    >>> processed_logits = processor(input_ids=inputs['input_ids'], scores=logits)

    >>> gen_out = model.generate(inputs['input_ids'], logits_processor=LogitsProcessorList([processor]))
    """

    def __init__(self, list_of_token_lists: list[list[int]]):
        self.token_state_machine = TokenStateMachine(list_of_token_lists)

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        batch_size = input_ids.size(0)
        for i in range(batch_size):
            current_tokens = input_ids[i].tolist()
            next_valid_tokens = self.token_state_machine.get_next_tokens(current_tokens)

            # Create a mask for the invalid tokens in the current batch
            invalid_tokens_mask = torch.full_like(scores[i], -float("inf"))
            for token in next_valid_tokens:
                invalid_tokens_mask[token] = 0

            # Apply the mask to the scores of the current batch
            scores[i] += invalid_tokens_mask

        return scores
