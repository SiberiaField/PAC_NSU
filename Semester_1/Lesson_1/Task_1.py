import random

a = random.randint(100, 999)
res = 0
while a > 0:
    res += int(a % 10)
    a /= 10
print(res)