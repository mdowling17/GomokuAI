import numpy as np
import numpy as np

original_list = np.array([1, 0, 1, 0, 0, 0, 1, 1, 0], dtype=int)
r = 0
c = 2
win_size = 5

base = original_list[c:c+win_size]

prefix, suffix = [], []
# suffix = []
for i in range(1, 3):
  if c-i < 0:
    break
  if original_list[c-i] == 1:
    prefix.append(1)
  else: break

for i in range(1, 3):
  if c+win_size-1+i >= len(original_list):
    break
  if original_list[c+win_size-1+i] == 1:
    suffix.append(1)
  else: break
print(prefix, base, suffix)
combined = np.array(np.append(np.append(prefix, base), suffix), dtype=int)
print(combined)
print("Original List:", combined)
inverted_list = np.array([bit ^ 1 for bit in combined], dtype=int)
print("Inverted List:", inverted_list)

assert(np.all(inverted_list == np.array([0,1,1,1,0,0],dtype=int)))