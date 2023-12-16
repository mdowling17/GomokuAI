import math
import numpy as np

"""
Implement your AI here
Do not change the API signatures for __init__ or __call__
__call__ must return a valid action
"""
NUM_ROLLOUTS = 100
UCT_EXPLORATION_FACTOR = math.sqrt(2)

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent

        self.total_visits = 0
        self.total_value = 0
        self.children = None
        # self.score_ema = 0. # exponential moving average

    def get_children(self):
        valid_actions = self.state.valid_actions()
        if self.children is None and len(valid_actions) != 0:
            self.children = [
                Node(self.state.perform(action), parent=self)
                for action in valid_actions
            ]
        return self.children

    def is_leaf(self):
        return self.children is None

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
        # add all valid actions as children and return a random child
        children = self.get_children()
        child = children[np.random.randint(len(children))]
        copy = child.copy()
        return copy

    def rollout(self):
        # randomly select a child and perform a rollout from that node
        # return the score of the rollout
        if self.state.is_game_over():
            return self.state.current_score()
        else:
            children = self.get_children()
            child = children[np.random.randint(len(children))]
            return child.rollout()

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


def mcts(state, num_rollouts):
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
        print(f"Rollout {i}")
        selected_node = root

        # selection
        while not selected_node.is_leaf():
            selected_node = selected_node.select()

        # expansion and simulation
        score = 0.0
        if selected_node.state.is_game_over():
            score = selected_node.state.current_score()
        else:
            random_child = selected_node.expand() # adds children to selected_node and returns a copy of a random child
            score = random_child.rollout() # return the terminal score of the rollout
        
        # backpropagation
        selected_node.backpropagate(score)
        print(f"Score: {score}, root visits: {root.total_visits}")

    # return the action with the highest visit count
    children = root.get_children()
    visits = [child.total_visits for child in children]
    best_child_index = np.argmax(visits)
    best_action = state.valid_actions()[best_child_index]
    # print rollouts, and best child action
    return best_action


class Submission:
    def __init__(self, board_size, win_size, num_rollouts=NUM_ROLLOUTS):
        self.num_rollouts = num_rollouts
        ### Add any additional initiation code here
        pass

    def __call__(self, state):
        action = mcts(state, self.num_rollouts)
        ### Replace with your implementation
        return action
