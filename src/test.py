import numpy as np

test = ((2,3),(1,2),(1,2),(1,2,),(2,4),(2,4))

arr = []

arr.extend(test)

# check if arr contains duplicates and if so print the first duplicate
seen = set()
uniq = []
for x in arr:
    if x not in seen:
        uniq.append(x)
        seen.add(x)
    else:
        print(x)
        break

# get the count of the duplicate
print(arr.count(x))