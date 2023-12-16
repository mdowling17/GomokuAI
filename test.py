import numpy as np

board_size = 15
win_size = 5
EMPTY = 0
board = np.zeros((3, board_size, board_size))


valid_actions = ((1, 2), (3, 4))
children = None
if children is None and len(valid_actions) != 0:     
    children = [action for action in valid_actions]
print(children)

print("end")