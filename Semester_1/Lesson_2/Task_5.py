def fib(n):
    if n < 1: return 0
    if n == 1: return 1
    return fib(n - 1) + fib(n - 2)
print(fib(2))
print(fib(4))
print(fib(10))