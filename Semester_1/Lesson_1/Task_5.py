b = int(input())

arr = [1] * (b + 1)
arr[0] = 0
for i in range(2, int(b / 2)):
    if(arr[i] == 1):
        for j in range(i + i, b + 1, i):
            arr[j] = 0

res = 0
for i in range(1, b + 1):
    res += arr[i]
print(res)