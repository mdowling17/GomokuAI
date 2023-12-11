"""
Implement your AI here
Do not change the API signatures for __init__ or __call__
__call__ must return a valid action
"""
class Submission:
    def __init__(self, board_size, win_size):
        ### Add any additional initiation code here
        pass

    def __call__(self, state):

        ### Replace with your implementation
        actions = state.valid_actions()
        return actions[-1]
    
