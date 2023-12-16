## Implementing MCTS:
First, MCTS receives a given state and then performs rollouts creates a root node.
Then, it performs successive rollouts on the tree.
A rollout consists of the following:
1. Selection: Perform a tree search using UCT until you reach a leaf node. Check whether a node is a leaf by if it currently does not have children. Split ties randomly.

2. Expansion: 
If the leaf.state.is_game_over() is True, then return leaf.state.current_score(). Else, get leaf.state.valid_actions from that state and perform each, creating new nodes and appending them to the children list of the leaf. (There may be room to optimize here by not doing this until needed). Then, call a rollout from a random one of its children.

3. Simulation
    * Perform a uniform random set of moves from that child until a game over state is reached. Return the score.
2. Backpropagation
    * Propagate the score updates from the rollout up the tree and increment visit counts until you reach the **root**
 
At the end of all the rollouts, you should select the action with the most visits.

Considerations:
* Saving runtime as much as possible by memoizing
* How do you store the data?

### state
`state` is an instance of the `GomokuState` class.

#### methods
* current_player() -> Int: returns the current player (1 for min 2 for max) in this state by counting bits in the layers of the board. If the x’s and o’s are equal, it’s the max player’s turn. Else it’s the min player’s turn. Caches result in state.player
* is_max_turn() -> Boolean: returns boolean whether current player is max; calls current_player under the hood
* current_score() -> Int: returns the current score in this state; checks for wins using self.corr. 0 if there’s a win on the board, else the count of empty spaces + 1. Negative if winner is min else positive.
* is_game_over() -> Boolean: returns boolean whether game is over in current state and caches result in state.over.
* valid_actions() -> tuple(tuple(r,c)): returns a tuple of (r, c) tuples, one for each empty position on the board.
* perform(action) -> GomokuState: returns a new state with the new position and updates the corr matrix
* copy() -> GomokuState
* blank(board_size, win_size) -> GomokuState
* play_seq(actions (list(GomokuState)), midgame (Boolean)) -> GomokuState

#### attributes
* state.board (GomokuState): one-hot, 3x15x15 matrix. The first slice is for empty spaces, then the min player's O's and lastly the max player's X's.
* state.win_size (Int): int = 5
* state.corr (np.ndarray (Shape: 4x3x15x15)): ??? progressively builds after each perform()
* state.player (Int): updated via current_player()
* state.score (Int): updated via current_score()
* state.over (Boolean): updated via is_game_over()
* state.actions (tuple(tuple(r,c))): updated via valid_actions()


## python syntax
.copy() on numpy arrays is a deep copy and changes to the copy do not modify the original

## improvement ideas
* use numpy arrays for everything


## considerations
* during the rollout (simulation) phase I need to make a copy so that rolled out children are not included in the tree.