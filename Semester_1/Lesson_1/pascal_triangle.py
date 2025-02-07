import argparse

def count_pascal(h):
    res = [[1]]
    for i in range(1, h + 1):
        res.append([1])
        for j in range(1, i):
            res[i].append(res[i - 1][j - 1] + res[i - 1][j])
        res[i].append(1)
    return res

def print_pascal(pascal_triangle, h):
    last_level = ' '.join(str(num) for num in pascal_triangle[h])
    mid = int(len(last_level) / 2)
    for i in range(0, h):
        print_level = ' '.join(str(num) for num in pascal_triangle[i])
        shift = mid - int(len(print_level) / 2)
        print(' ' * shift, print_level, sep='')
    print(last_level)
    

parser = argparse.ArgumentParser(description = "Make pascal triangle with height H")
parser.add_argument("height", type = int, nargs= "?", default = 3, metavar= "H", 
                    help = "height of pascal triangle, H >= 0, int, default = 3")

H = parser.parse_args().height
if H < 0:
    print("\nInvalid input: H < 0\n")
    exit(1)

print_pascal(count_pascal(H), H)

