class StateMachineNode:
    def __init__(self, token_id: int):
        self.token_id = token_id
        self.children = {}


class TokenStateMachine:
    def __init__(
            self,
            no_prefix_space_word_token_lists: list[list[int]],
            prefix_space_word_token_lists: list[list[int]],
            special_token_lists: list[list[int]],
        ):
        self.root = StateMachineNode(-1)
        self.root_prefix_space_special = StateMachineNode(-1)
        for token_list in no_prefix_space_word_token_lists:
            self._add_token_list(token_list, self.root, self.root_prefix_space_special)
        for token_list in prefix_space_word_token_lists:
            self._add_token_list(token_list, self.root, self.root_prefix_space_special)
            self._add_token_list(token_list, self.root_prefix_space_special, self.root_prefix_space_special)
        for token_list in special_token_lists:
            self._add_token_list(token_list, self.root, self.root)
            self._add_token_list(token_list, self.root_prefix_space_special, self.root)

    def _add_token_list(self, token_list: list[int], root: StateMachineNode, end_root: StateMachineNode):
        current_node = root
        for token in token_list:
            if token not in current_node.children:
                current_node.children[token] = StateMachineNode(token)
            current_node = current_node.children[token]
        current_node.children[-1] = end_root

    def get_next_tokens(self, tokens: list[int]) -> list[int]:
        """Get the next possible tokens given a sequence of tokens."""
        current_node = self.root
        for token in tokens:
            if token in current_node.children:
                current_node = current_node.children[token]
            elif -1 in current_node.children:
                current_node = current_node.children[-1]
                if token in current_node.children:
                    current_node = current_node.children[token]
            else:
                current_node = self.root
                if token in current_node.children:
                    current_node = current_node.children[token]
        next_tokens = list(current_node.children.keys())
        if not next_tokens:
            next_tokens = list(self.root.children.keys())
        elif -1 in next_tokens:
            next_tokens.extend(current_node.children[-1].children.keys())
        next_tokens = [token for token in next_tokens if token != -1]
        return next_tokens
