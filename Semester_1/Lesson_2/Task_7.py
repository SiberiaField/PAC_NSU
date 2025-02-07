def geo_prog(q, b1):
    n = 1
    while True:
        n += 1
        yield b1 * q**(n - 1)

answ = geo_prog(2, 3)
for i in range(4):
    print(next(answ))