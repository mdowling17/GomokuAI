"""
Implement your AI here
Do not change the API signatures for __init__ or __call__
__call__ must return a valid action
"""

# ====================================

import itertools as it
import numpy as np
import gomoku as gm
import time
from policies.lookup import lookup

MAX_DEPTH=225
TIME_LIMIT=0.1
# helper function to get minimal path length to a game over state
# @profile
def turn_bound(state):

    is_max = state.is_max_turn()
    fewest_moves = state.board[gm.EMPTY].sum() # moves to a tie game

    # use correlations to extract possible routes to a non-tie game
    corr = state.corr
    min_routes = (corr[:,gm.EMPTY] + corr[:,gm.MIN] == state.win_size)
    max_routes = (corr[:,gm.EMPTY] + corr[:,gm.MAX] == state.win_size)
    # also get the number of turns in each route until game over
    min_turns = 2*corr[:,gm.EMPTY] - (0 if is_max else 1)
    max_turns = 2*corr[:,gm.EMPTY] - (1 if is_max else 0)

    # check if there is a shorter path to a game-over state
    if min_routes.any():
        moves_to_win = min_turns.flatten()[min_routes.flatten()].min()
        fewest_moves = min(fewest_moves, moves_to_win)
    if max_routes.any():
        moves_to_win = max_turns.flatten()[max_routes.flatten()].min()
        fewest_moves = min(fewest_moves, moves_to_win)

    # return the shortest path found to a game-over state
    return fewest_moves

# helper to find empty position in pth win pattern starting from (r,c)
def find_empty(state, p, r, c):
    if p == 0: # horizontal
        return r, c + state.board[gm.EMPTY, r, c:c+state.win_size].argmax()
    if p == 1: # vertical
        return r + state.board[gm.EMPTY, r:r+state.win_size, c].argmax(), c
    if p == 2: # diagonal
        rng = np.arange(state.win_size)
        offset = state.board[gm.EMPTY, r + rng, c + rng].argmax()
        return r + offset, c + offset
    if p == 3: # antidiagonal
        rng = np.arange(state.win_size)
        offset = state.board[gm.EMPTY, r - rng, c + rng].argmax()
        return r - offset, c + offset
    # None indicates no empty found
    return None

def get_position_list(lhs, rhs, base):
    base = np.array(base, dtype=int)
    l = lhs - 1
    r = len(base) - rhs
    while l >= 0:
      if base[l] != 1: break
      l -= 1
    while r < len(base):
      if base[r] != 1: break
      r += 1
    l += 1

    slice = base[l:r+1]
    inverted_list = np.array([bit ^ 1 for bit in slice], dtype=int)
    return inverted_list, lhs-l

def find_winning_patterns(state, p, r, c):
    if p == 0: # horizontal
        min_c = max(0, c-2)
        max_c = min(state.board.shape[2]-1, c+state.win_size+1)
        base = state.board[gm.EMPTY, r, min_c:max_c+1]
        lhs = c-min_c
        rhs  = max_c-(c+state.win_size-1)
        position, lhs = get_position_list(lhs, rhs, base)
        lookup_val = lookup(position)
        if lookup_val is None: return None
        index = (lookup_val - lhs)+c
        return r, index
    if p == 1: # vertical
        min_r = max(0, r-2)
        max_r = min(state.board.shape[1]-1, r+state.win_size+1)
        base = state.board[gm.EMPTY, min_r:max_r+1, c]
        lhs = r-min_r
        rhs  = max_r-(r+state.win_size-1)
        position, lhs = get_position_list(lhs, rhs, base)
        lookup_val = lookup(position)
        if lookup_val is None: return None
        index = (lookup_val - lhs)+r
        return index, c
    if p == 2: # diagonal
        min_r = max(0, r-min(2, c))
        min_c = max(0, c-min(2, r))
        max_r = min(state.board.shape[1]-1, r+min(state.win_size+1, state.board.shape[2]-(c+state.win_size-1)-1))
        max_c = min(state.board.shape[2]-1, c+min(state.win_size+1, state.board.shape[1]-(r+state.win_size-1)-1))
        max_len = min(max_r-min_r, max_c-min_c) + 1
        rng = np.arange(max_len)
        base = state.board[gm.EMPTY, min_r + rng, min_c + rng]
        lhs = min(r-min_r, c-min_c)
        rhs = min(max_r-(r+state.win_size-1), max_c-(c+state.win_size-1))
        position, lhs = get_position_list(lhs, rhs, base)
        lookup_val = lookup(position)
        if lookup_val is None: return None
        updated_r = (lookup_val - lhs) + r
        updated_c = (lookup_val - lhs) + c
        return updated_r, updated_c
    if p == 3: # anti-diagonal
        min_r = max(0, r-(state.win_size-1)-min(2, state.board.shape[2]-(c+state.win_size-1)-1))
        min_c = max(0, c-min(2, state.board.shape[1]-r-1))
        max_r = min(state.board.shape[1]-1, r+min(2, c))
        max_c = min(state.board.shape[2]-1, c+state.win_size-1+min(2, r-(state.win_size-1)))
        max_len = min(max_r-min_r, max_c-min_c) + 1
        rng = np.arange(max_len)
        base = state.board[gm.EMPTY, max_r - rng, min_c + rng]
        lhs = min(max_r-r, c-min_c)
        rhs = min((r-state.win_size+1)-min_r, max_c-(c+state.win_size-1))
        position, lhs = get_position_list(lhs, rhs, base)
        lookup_val = lookup(position)
        if lookup_val is None: return None
        updated_r = r - (lookup_val - lhs)
        updated_c = (lookup_val - lhs) + c
        return updated_r, updated_c

def opponent_wins_in_one(state):
    player = state.current_player()
    opponent = gm.MIN if player == gm.MAX else gm.MAX
    corr = state.corr
    # check if opponent is one move away to a win
    idx = np.argwhere((corr[:, gm.EMPTY] == 1) & (corr[:, opponent] == state.win_size-1))
    actions = []
    for p, r, c in idx:
        actions.append(find_empty(state, p, r, c))
    if len(set(actions)) == 1:
        action = actions[0]
        return action
    elif len(set(actions)) > 1:
        return False
    return None

def look_ahead(state):

    # if current player has a win pattern with all their marks except one empty, they can win next turn
    player = state.current_player()
    opponent = gm.MIN if player == gm.MAX else gm.MAX
    sign = +1 if player == gm.MAX else -1
    magnitude = state.board[gm.EMPTY].sum() # no +1 since win comes after turn

    # check if current player is one move away to a win
    corr = state.corr
    idx = np.argwhere((corr[:, gm.EMPTY] == 1) & (corr[:, player] == state.win_size-1))
    if idx.shape[0] > 0:
        # find empty position they can fill to win, it is an optimal action
        p, r, c = idx[0]
        action = find_empty(state, p, r, c)
        return sign * magnitude, action

    # else, if opponent has at least two such moves with different empty positions, they can win in two turns
    loss_empties = set() # make sure the 2+ empty positions are distinct
    idx = np.argwhere((corr[:, gm.EMPTY] == 1) & (corr[:, opponent] == state.win_size-1))
    for p, r, c in idx:
        pos = find_empty(state, p, r, c)
        assert pos is not None
        loss_empties.add(pos)        
        if len(loss_empties) > 1: # just found a second empty
            score = -sign * (magnitude - 1) # opponent wins an extra turn later
            return score, pos # block one of their wins with next action even if futile
    
    # ===================================
    # === Further Modifications Below ===
    # ===================================
    # covered patterns: XXX & X_XX including their mirror images, translations with various empty-space prefixes and suffixes
    # if the player has a particular 3-piece pattern they can force a win in 2 turns as long as the opponent can't win in 4 turns
    idx = np.argwhere((corr[:, gm.EMPTY] == 2) & (corr[:, player] == state.win_size-2))
    for p, r, c in idx:
        match = find_winning_patterns(state, p, r, c)
        if match is not None:
            if match not in state.valid_actions():
                print(f"match not in valid actions: {match}")
                return 0, None
            # win in 2 turns
            need_to_block_opponent = opponent_wins_in_one(state)
            if need_to_block_opponent is None or need_to_block_opponent == match:
              score = sign * (magnitude - 2)
              return score, match

    # return 0 to signify no conclusive look-aheads
    return 0, None

# recursive minimax search with additional pruning
# @profile
def minimax(state, max_depth, time_limit=None, alpha=-np.inf, beta=np.inf):
    # check for game over base case with no valid actions
    if state.is_game_over():
        return state.current_score(), None
    
    # check fast look-ahead before trying minimax
    score, action = look_ahead(state)
    if score != 0: return score, action
  
    # have to try minimax, prepare the valid actions
    # should be at least one valid action if this code is reached
    actions = state.valid_actions()

    # prioritize actions near non-empties but break ties randomly
    rank = -state.corr[:, 1:].sum(axis=(0,1)) - np.random.rand(*state.board.shape[1:])
    rank = rank[state.board[gm.EMPTY] > 0] # only empty positions are valid actions
    scrambler = np.argsort(rank)

    # check for max depth base case
    if max_depth == 0 or (time_limit is not None and time.time() > time_limit):
        return state.current_score(), actions[scrambler[0]]

    # custom pruning: stop search if no path from this state wins within max_depth turns
    if turn_bound(state) > max_depth: return 0, actions[scrambler[0]]

    # alpha-beta pruning
    best_action = None
    if state.is_max_turn():
        bound = -np.inf
        for a in scrambler:
            action = actions[a]
            child = state.perform(action)
            utility, _ = minimax(child, max_depth-1, time_limit=time_limit, alpha=alpha, beta=beta)

            if utility > bound: bound, best_action = utility, action
            if bound >= beta: break
            alpha = max(alpha, bound)

    else:
        bound = +np.inf
        for a in scrambler:
            action = actions[a]
            child = state.perform(action)
            utility, _ = minimax(child, max_depth-1, time_limit=time_limit, alpha=alpha, beta=beta)

            if utility < bound: bound, best_action = utility, action
            if bound <= alpha: break
            beta = min(beta, bound)

    return bound, best_action

# Policy wrapper
class Submission:
    def __init__(self, board_size, win_size):
        self.max_depth = MAX_DEPTH


    def __call__(self, state):
        # set a time limit for the minimax search equal to 0.1 seconds from now
        time_limit = time.time() + TIME_LIMIT
        # run iterative deepening minimax search
        for i in range(0, self.max_depth+1):
            if i <= 4: _, action = minimax(state=state, max_depth=i, time_limit=None)
            else:
                if time.time() > time_limit: break
                _, action = minimax(state=state, max_depth=i, time_limit=time_limit)
        return action

if __name__ == "__main__":

    # unit tests for look-ahead function

    state = gm.GomokuState.blank(5, 3)
    state = state.play_seq([(0,0), (0,1), (1,1), (1,2)])
    score, action = look_ahead(state)
    assert score == 1 + 5**2 - 5
    assert action == (2,2)

    state = gm.GomokuState.blank(5, 3)
    state = state.play_seq([(4,1), (4,2), (3,2), (3,3)])
    score, action = look_ahead(state)
    assert score == 1 + 5**2 - 5
    assert action == (2,3)


    # =============================================
    # === Test horizontal find_winning_patterns ===
    # =============================================
    # print 10 newlines
    print("\n"*10)

    state = gm.GomokuState.blank(15, 5)
    state = state.play_seq([(0,0), (7,7), (14,0), (7,8), (0,14), (7,9), (14,14)])
    corr = state.corr
    player = state.current_player()
    assert player == gm.MIN
    
    sign = +1 if player == gm.MAX else -1
    magnitude = state.board[gm.EMPTY].sum()
    print(state)
    idx = np.argwhere((corr[:, gm.EMPTY] == 2) & (corr[:, player] == state.win_size-2))
    assert len(idx) != 0

    for p, r, c in idx:
        print(p, r, c)
        match = find_winning_patterns(state, p, r, c)
        if match is not None:
            # win in 2 turns
            score = sign * (magnitude - 2)
            
            print(f"score: {score}, match: {match}")

    # =============================================
    # === Test vertical find_winning_patterns ===
    # =============================================
    # print 10 newlines
    print("\n"*10)

    state = gm.GomokuState.blank(15, 5)
    state = state.play_seq([(0,0), (7,7), (14,0), (8,7), (0,14), (9,7), (14,14)])
    corr = state.corr
    player = state.current_player()
    assert player == gm.MIN
    
    sign = +1 if player == gm.MAX else -1
    magnitude = state.board[gm.EMPTY].sum()
    print(state)
    idx = np.argwhere((corr[:, gm.EMPTY] == 2) & (corr[:, player] == state.win_size-2))
    assert len(idx) != 0

    for p, r, c in idx:
        print(p, r, c)
        match = find_winning_patterns(state, p, r, c)
        if match is not None:
            # win in 2 turns
            score = sign * (magnitude - 2)
            
            print(f"score: {score}, match: {match}")

    # =============================================
    # === Test diagonal find_winning_patterns ===
    # =============================================
    # print 10 newlines
    print("\n"*10)

    state = gm.GomokuState.blank(15, 5)
    state = state.play_seq([(0,0), (7,7), (14,0), (8,8), (0,14), (9,9), (14,14)])
    corr = state.corr
    player = state.current_player()
    assert player == gm.MIN
    
    sign = +1 if player == gm.MAX else -1
    magnitude = state.board[gm.EMPTY].sum()
    print(state)
    idx = np.argwhere((corr[:, gm.EMPTY] == 2) & (corr[:, player] == state.win_size-2))
    assert len(idx) != 0

    for p, r, c in idx:
        print(p, r, c)
        match = find_winning_patterns(state, p, r, c)
        if match is not None:
            # win in 2 turns
            score = sign * (magnitude - 2)
            
            print(f"score: {score}, match: {match}")

    # ================================================
    # === Test anti-diagonal find_winning_patterns ===
    # ================================================
    # print 10 newlines
    print("\n"*10)

    state = gm.GomokuState.blank(15, 5)
    state = state.play_seq([(0,0), (7,7), (14,0), (6,8), (0,14), (5,9), (14,14)])
    corr = state.corr
    player = state.current_player()
    assert player == gm.MIN
    
    sign = +1 if player == gm.MAX else -1
    magnitude = state.board[gm.EMPTY].sum()
    print(state)
    idx = np.argwhere((corr[:, gm.EMPTY] == 2) & (corr[:, player] == state.win_size-2))
    assert len(idx) != 0

    for p, r, c in idx:
        print(p, r, c)
        match = find_winning_patterns(state, p, r, c)
        if match is not None:
            # win in 2 turns
            score = sign * (magnitude - 2)
            
            print(f"score: {score}, match: {match}")

    # ================================================
    # === Test diagonal with 1 space and 2 spaces ===
    # ================================================
    # print 10 newlines
    print("\n"*10)

    state = gm.GomokuState.blank(15, 5)
    state = state.play_seq([(4,0), (6,2), (14,0), (7,3), (0,14), (8,4), (14,14)])
    corr = state.corr
    player = state.current_player()
    assert player == gm.MIN
    
    sign = +1 if player == gm.MAX else -1
    magnitude = state.board[gm.EMPTY].sum()
    print(state)
    idx = np.argwhere((corr[:, gm.EMPTY] == 2) & (corr[:, player] == state.win_size-2))
    assert len(idx) != 0
    print(idx)
    for p, r, c in idx:
        print(p, r, c)
        match = find_winning_patterns(state, p, r, c)
        if match is not None:
            # win in 2 turns
            score = sign * (magnitude - 2)
            assert match in state.valid_actions()
            print(f"score: {score}, match: {match}")
    print("no fails")

    # ================================================
    # === Test anti-diagonal with 1 space and 2 spaces ===
    # ================================================
    # print 10 newlines
    print("\n"*10)

    state = gm.GomokuState.blank(15, 5)
    state = state.play_seq([(11,0), (9,2), (14,0), (8,3), (0,14), (7,4), (14,14)])
    corr = state.corr
    player = state.current_player()
    assert player == gm.MIN
    
    sign = +1 if player == gm.MAX else -1
    magnitude = state.board[gm.EMPTY].sum()
    print(state)
    idx = np.argwhere((corr[:, gm.EMPTY] == 2) & (corr[:, player] == state.win_size-2))
    assert len(idx) != 0
    print(idx)
    for p, r, c in idx:
        print(p, r, c)
        match = find_winning_patterns(state, p, r, c)
        if match is not None:
            # win in 2 turns
            score = sign * (magnitude - 2)
            assert match in state.valid_actions()
            print(f"score: {score}, match: {match}")
    print("no fails")

