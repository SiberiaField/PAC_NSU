import argparse

def read_matrix(text: list[str]):
    matrix = []
    i = 0
    for str in text:
        matrix.append([])
        for num in str.split():
            matrix[i].append(int(num))
        i += 1
    return matrix

parser = argparse.ArgumentParser(description = "Matrix convolution")
parser.add_argument('files', metavar= 'input and ouput files', type= str, nargs= 2)

files = parser.parse_args().files       
matrices = open(files[0], 'r').read().split("\n\n")
first_part = matrices[0].split('\n')
second_part = matrices[1].split('\n')
matrix_A = read_matrix(first_part)
matrix_B = read_matrix(second_part)

lines_A = len(matrix_A)
collums_A = len(matrix_A[0])
lines_B = len(matrix_B)
collums_B = len(matrix_B[0])
'''print(f"lines_A: {lines_A} collums_A: {collums_A}")
print(f"lines_B: {lines_B} collums_B: {collums_B}")'''

def dot_prod(beg: list) -> int:
    res = 0
    k = 0; i = beg[0]
    while i < beg[0] + lines_B:
        l = 0; j = beg[1]
        while j < beg[1] + collums_B:
            res += matrix_B[k][l] * matrix_A[i][j]
            j += 1; l += 1
        i += 1; k += 1
    return res

def matrix_conv() -> str:
    if lines_A >= lines_B and collums_A >= collums_B:
        res = ''
        lines_res = lines_A - lines_B + 1
        collums_res = collums_A - collums_B + 1
        for i in range(0, lines_res):
            for j in range(0, collums_res):
                res += str(dot_prod([i, j])) + ' '
            res += '\n'
        return res
    else: return ''

output = open(files[1], 'w')
res = matrix_conv()
if res == '':
    output.writelines("Error: sizes of matrices are not correct")
else:
    output.writelines(res)
            

