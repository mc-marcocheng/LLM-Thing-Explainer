import string

from transformers import PreTrainedTokenizer


def create_token_lists(
        tokenizer: PreTrainedTokenizer,
        words: list[str],
        add_punctuations: bool = True,
        add_numbers: bool = True,
    ) -> list[list[int]]:
    """
    Create token lists from words using a tokenizer, optionally adding punctuations and numbers.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding the words.
        words (list[str]): The list of words to tokenize.
        add_punctuations (bool, optional): Whether to add punctuation tokens to the list. Defaults to True.
        add_numbers (bool, optional): Whether to add number tokens to the list. Defaults to True.

    Returns:
        list[list[int]]: A list of token lists, where each sublist is a sequence of token IDs.
    """

    token_lists = []
    dummy_first_token = tokenizer.convert_ids_to_tokens(0)
    prefix_space_char = tokenizer.convert_ids_to_tokens(tokenizer.encode(dummy_first_token + ' ' + dummy_first_token, add_special_tokens=False)[-1])[0]

    # Make title case words
    words.extend(list(map(str.title, words)))
    # Add punctuations
    if add_punctuations:
        words.extend(string.punctuation)
    # Add numbers
    if add_numbers:
        for vocab in tokenizer.vocab:
            if vocab.isdigit() or vocab.lstrip(prefix_space_char).isdigit():
                words.append(vocab)

    # Without prefix space tokens
    token_lists.extend([tokenizer.encode(word, add_special_tokens=False) for word in words])
    # With prefix space tokens
    token_lists.extend([tokenizer.encode(dummy_first_token + ' ' + word, add_special_tokens=False)[1:] for word in words])
    # Add special tokens
    token_lists.extend([[i] for i in tokenizer.all_special_ids])
    for added_token in tokenizer.get_added_vocab().values():
        token_lists.append([added_token])

    return token_lists
