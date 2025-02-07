import random
import sys

a = random.randint(0, sys.maxsize)
res = 0
while a > 0:
    res += int(a % 10)
    a /= 10
print(res)