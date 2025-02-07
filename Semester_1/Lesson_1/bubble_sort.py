import argparse
import random

parser = argparse.ArgumentParser(description = "Bubble sort")
parser.add_argument('arr_size', metavar = 'N', type = int, nargs = '?', default = 2,
                    help = "size of sorting arr, N > 0, int, default = 2")

N = parser.parse_args().arr_size
if N < 1: 
    print("\nInvalid input: N < 1\n")
    exit(1)

arr = []
for i in range(0, N):
    arr.append(random.random())
print("\nArray:",arr)
print("Sorting...")
for i in range(0, N):
    for j in range(i + 1, N):
        if arr[i] > arr[j]:
            arr[i], arr[j] = arr[j], arr[i]
print("Result:",arr,"\n")