import random

N = int(input())
arr = [random.randint(-1000, 1000) for i in range(0, N)]
print("Arr:", *arr)

even = 0
odd = 0
for num in arr:
    if num % 2 == 0:
        even += 1
    else:
        odd += 1
print("Even nums:", even)
print("Odd nums:", odd)
