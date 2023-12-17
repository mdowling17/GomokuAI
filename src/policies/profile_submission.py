import cProfile
from src.policies.submission import Submission
import gomoku as gm

def profile_submission():
    # Setup your state and num_rollouts here
    state = gm.GomokuState.blank(15, 5)
    state = state.play_seq([(1,7)])
    print(state)
    
    policy = Submission(15, 5)


    profiler = cProfile.Profile()
    profiler.enable()

    action = policy(state)
    print(action)
    state = state.perform(action)
    print(state)

    profiler.disable()
    profiler.print_stats(sort='time')

profile_submission()