from itertools import combinations, product
import numpy as np
# number of empty spaces
n_empty = 5

# number of player markers
n_player = 3

# generate all combinations of positions 
combinations = list(combinations(range(n_empty), n_player))

# print each combination as a string of 1s (player markers) and 0s (empty spaces)
unpadded_list = []
for combo in combinations:
    arr = [0] * n_empty
    for pos in combo:
        arr[pos] = 1
    unpadded_list.append(arr)

# print the number of combinations
# print(f"Number of combinations: {len(combinations)}")

# ================================================================
# adding padding

# possible prefixes and suffixes
prefixes_suffixes = [[], [0], [0, 0]]

# generate all combinations of prefixes and suffixes
combinations = list(product(prefixes_suffixes, repeat=2))

# create a new list that consists of the combinations with the prefixes and suffixes
padded_list = []
for (prefix, suffix) in combinations:
    len_prefix = len(prefix)
    len_suffix = len(suffix)
    if len_prefix + len_suffix == 0: continue
    for combo in unpadded_list:
        padded_combo = prefix + combo + suffix
        index = None
        zeros = [i for i, n in enumerate(combo) if n == 0]
        if combo == [0, 1, 1, 1, 0]:
            if len_prefix > 0:
              index = len_prefix + zeros[0]
            elif len_suffix > 0:
              index = len_prefix + zeros[1]
        elif combo == [1, 1, 1, 0, 0] and len_prefix > 0:
            index = len_prefix + zeros[0]
        elif combo == [0, 0, 1, 1, 1] and len_suffix > 0:
            index = len_prefix + zeros[1]
        elif combo == [1, 0, 1, 1, 0] and len_prefix > 0:
            index = len_prefix + zeros[0]
        elif combo == [1, 1, 0, 1, 0] and len_prefix > 0:
            index = len_prefix + zeros[0]
        elif combo == [0, 1, 1, 0, 1] and len_suffix > 0:
            index = len_prefix + zeros[1]
        elif combo == [0, 1, 0, 1, 1] and len_suffix > 0:
            index = len_prefix + zeros[1]
          
        padded_list.append({tuple(padded_combo): index})

# # print the new list
# for combo in padded_list:
#     print(combo)

# print(f"length padded_list: {len(padded_list)}")

# create a new dictionary from the list of key-value pairs in padded_list
hash_table = {}
for combo in padded_list:
    hash_table.update(combo)


# function to look up a value for a list of 0's and 1's in the hash table
def lookup(lst):
    return hash_table.get(tuple(lst), None)  # returns None if lst is not in the hash table

# example usage
# print(lookup([0, 1, 1, 1, 0, 0, 0]))
