class StateMachineNode:
    def __init__(self, token_id: int):
        self.token_id = token_id
        self.children = {}


class TokenStateMachine:
    def __init__(self, list_of_token_lists: list[list[int]]):
        self.root = StateMachineNode(-1)
        for token_list in list_of_token_lists:
            self._add_token_list(token_list)

    def _add_token_list(self, token_list: list[int]):
        current_node = self.root
        for token in token_list:
            if token not in current_node.children:
                current_node.children[token] = StateMachineNode(token)
            current_node = current_node.children[token]

    def get_next_tokens(self, tokens: list[int]) -> list[int]:
        """Get the next possible tokens given a sequence of tokens."""
        current_node = self.root
        for token in tokens:
            if token in current_node.children:
                current_node = current_node.children[token]
            else:
                current_node = self.root
                if token in current_node.children:
                    current_node = current_node.children[token]
        return list(current_node.children.keys()) + list(self.root.children.keys())
