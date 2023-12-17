import cProfile
from submission import mcts
import gomoku as gm

def profile_mcts():
    # Setup your state and num_rollouts here
    state = gm.GomokuState.blank(15, 5)
    state = state.perform((1,7))
    print(state)

    profiler = cProfile.Profile()
    profiler.enable()

    mcts(state)

    profiler.disable()
    profiler.print_stats(sort='time')

profile_mcts()