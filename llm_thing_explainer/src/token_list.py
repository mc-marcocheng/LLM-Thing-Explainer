import string

from transformers import PreTrainedTokenizer


def create_token_lists(
        tokenizer: PreTrainedTokenizer,
        words: list[str],
        add_punctuations: bool = True,
        add_numbers: bool = True,
        add_added_vocab: bool = True,
    ) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    """
    Create token lists from words using a tokenizer, optionally adding punctuations and numbers.

    Args:
        tokenizer (PreTrainedTokenizer): The tokenizer to use for encoding the words.
        words (list[str]): The list of words to tokenize.
        add_punctuations (bool, optional): Whether to add punctuation tokens to the list. Defaults to True.
        add_numbers (bool, optional): Whether to add number tokens to the list. Defaults to True.
        add_added_vocab (bool, optional): Whether to add tokenizer.get_added_vocab() to the list.
            Useful for adding custom tokens like <think>. Defaults to True.

    Returns:
        tuple[list[list[int]], list[list[int]], list[list[int]]]:
            - A list of token lists without prefix spaces, where each sublist is a sequence of token IDs.
            - A list of token lists with prefix spaces, where each sublist is a sequence of token IDs.
            - A list of token lists for special tokens, where each sublist is a sequence of token IDs.
    """

    no_prefix_space_word_token_lists, prefix_space_word_token_lists, special_token_lists = [], [], []
    dummy_first_token = tokenizer.convert_ids_to_tokens(0)
    prefix_space_char = tokenizer.convert_ids_to_tokens(tokenizer.encode(dummy_first_token + ' ' + dummy_first_token, add_special_tokens=False)[-1])[0]

    # Make title case words
    words.extend(list(map(str.title, words)))

    special_chars = []
    # Add punctuations
    if add_punctuations:
        special_chars.extend(string.punctuation)
    # Add numbers
    if add_numbers:
        for vocab in tokenizer.vocab:
            if vocab.isdigit() or vocab.lstrip(prefix_space_char).isdigit():
                special_chars.append(vocab)

    # Without prefix space tokens
    no_prefix_space_word_token_lists.extend(tokenizer.encode(word, add_special_tokens=False) for word in words)
    special_token_lists.extend(tokenizer.encode(special_char, add_special_tokens=False) for special_char in special_chars)
    # With prefix space tokens
    prefix_space_word_token_lists.extend(tokenizer.encode(dummy_first_token + ' ' + word, add_special_tokens=False)[1:] for word in words)
    special_token_lists.extend(tokenizer.encode(dummy_first_token + ' ' + special_char, add_special_tokens=False)[1:] for special_char in special_chars)
    # Add special tokens
    special_token_lists.extend([i] for i in tokenizer.all_special_ids)

    # Add custom tokens
    if add_added_vocab:
        for added_token in tokenizer.get_added_vocab().values():
            special_token_lists.append([added_token])

    return no_prefix_space_word_token_lists, prefix_space_word_token_lists, special_token_lists
