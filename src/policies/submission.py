import math
import numpy as np
import gomoku as gm

"""
Implement your AI here
Do not change the API signatures for __init__ or __call__
__call__ must return a valid action
"""
NUM_ROLLOUTS = 1000
UCT_EXPLORATION_FACTOR = math.sqrt(2)

def rollout(state):
    # randomly select a child and perform a rollout from that node
    # return the score of the rollout
    if state.is_game_over():
        return state.current_score()
    else:
        valid_actions = state.valid_actions()
        random_action = valid_actions[np.random.randint(len(valid_actions))]
        return rollout(state.perform(random_action))
        

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent

        self.total_visits = 0
        self.total_value = 0
        self.children = []
        # self.score_ema = 0. # exponential moving average
    
    def __str__(self):
        """
        Return string representation of this state
        """
        # return the current node's state as a string and the first 3 children's states as strings
        return "\n\n".join([str(self.state), *[str(child.state) for child in self.get_children()[:3]]])
    
    def get_children(self):
        return self.children

    def is_selectable(self):
        return len(self.get_children()) != len(self.state.valid_actions())

    def select(self):
        # select child with max UCT
        # UCT = Q + c * sqrt(ln(N) / n)
        # Q = average score of child
        # N = total number of visits of parent
        # n = number of visits of child
        # c = exploration factor
        # c = sqrt(2)
        # perform numpy operations to get max UCT of the children
        N = self.total_visits
        children = self.get_children()
        n = [child.total_visits for child in children]
        
        # negate utilities for min player "O"
        sign = +1.0 if self.state.is_max_turn() else -1.0

        # special case to handle 0 denominator for never-visited children
        Q = [0.0 if child.total_visits == 0 else sign * child.total_value / child.total_visits for child in children]
        c = UCT_EXPLORATION_FACTOR
        UCT = Q + c * np.sqrt(max(N, 1) / np.maximum(n, 1))
        return children[np.argmax(UCT)]

    def expand(self):
        # pick a random valid action and append it to the children of the current node
        # return a copy of the child node's state
        valid_actions = self.state.valid_actions()
        random_action = valid_actions[np.random.randint(len(valid_actions))]
        child_state = self.state.perform(random_action)
        child = Node(child_state, parent=self)
        self.children.append(child)
        copy = child.state.copy()
        return copy

    def backpropagate(self, score):
        # update total visits and total value of all nodes in the path from the leaf node to the root node
        self.total_visits += 1
        self.total_value += score  # sign?
        if self.parent is not None:
            self.parent.backpropagate(score)

    def copy(self):
        # return a copy of the node
        copy = Node(self.state.copy(), parent=self.parent)
        copy.total_visits = self.total_visits
        copy.total_value = self.total_value
        copy.children = self.children
        return copy


def mcts(state, num_rollouts=NUM_ROLLOUTS):
    """
    Monte Carlo Tree Search
    Selection: perform tree search, selecting the child node with max UCT at each branch until you reach a leaf node.
    Expansion: append all the possible valid actions from that leaf node state as children of the leaf.
    Simulation: randomly select one of the child nodes and perform a rollout from that node. A rollout consists of randomly selecting actions until you reach a game over state.
    Backpropagation: update the score totals and visit counts of all the nodes in the path from the leaf node to the root node.
    Repeat this process num_rollouts times and return the child node (action) of the root state with the highest visit count.
    """
    root = Node(state)
    for i in range(num_rollouts):
        selected_node = root

        # selection
        while not selected_node.is_selectable():
            selected_node = selected_node.select()

        assert selected_node.is_selectable()
        # expansion and simulation
        score = 0.0
        if selected_node.state.is_game_over():
            score = selected_node.state.current_score()
        else:
            random_child_state = selected_node.expand() # adds children to selected_node and returns a copy of a random child
            score = rollout(random_child_state) # return the terminal score of the rollout
        
        # backpropagation
        selected_node.backpropagate(score)

    # return the action with the highest visit count
    children = root.get_children()
    visits = [child.total_visits for child in children]
    best_child_index = np.argmax(visits)
    best_action = state.valid_actions()[best_child_index]
    # print rollouts, and best child action
    return best_action


class Submission:
    def __init__(self, board_size, win_size):
        ### Add any additional initiation code here
        pass

    def __call__(self, state):
        action = mcts(state)
        ### Replace with your implementation
        return action

if __name__ == "__main__":
    # ===================================
    # === Unit Test for Node.__init__ ===
    # ===================================
    node = Node(None)

    # Check that the node has the correct properties
    assert node.state is None
    assert node.parent is None
    assert node.total_value == 0
    assert node.total_visits == 0
    assert node.children == []

    # ========================================
    # === Unit Tests for Node.get_children ===
    # ========================================
    state = gm.GomokuState.blank(15, 5)

    node = Node(state)
    children = node.get_children()
    assert len(children) == 0

    max_win_state = state.play_seq([(0,0), (0,1), (1,0), (1,1), (2,0), (2,1), (3,0), (3,1), (4,0)])

    winning_node = Node(max_win_state)
    assert winning_node.state.is_game_over() == True
    assert winning_node.state.current_score() == 1 + 15**2 - 9
    assert winning_node.get_children() == []
    
    # ====================================
    # === Unit Test for Node.is_leaf() ===
    # ====================================
    state = gm.GomokuState.blank(15, 5)
    state2 = state.perform((0,0))
    root = Node(state)
    node2 = Node(state2, parent=root)

    assert root.children == []
    assert root.is_selectable() == True

    root.children.append(node2)
    assert root.is_leaf() == False
    assert node2.is_selectable() == True
    # ====================================
    # === Unit Test for Node. ===
    # ====================================

    print("✅ no fails")