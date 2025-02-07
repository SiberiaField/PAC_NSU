import argparse

def read_matrix(text):
    matrix = []
    i = 0
    for str in text:
        matrix.append([])
        for num in str.split():
            matrix[i].append(int(num))
        i += 1
    return matrix

def matrix_mult(matrix_A, matrix_B):
    lines_A = len(matrix_A)
    collums_A = len(matrix_A[0])
    lines_B = len(matrix_B)
    collums_B = len(matrix_B[0])

    res = ''
    if collums_A == lines_B:
        for k in range(0, lines_A):
            for i in range(0, collums_B):
                new_elem = 0
                for j in range(0, lines_B):
                    new_elem += matrix_A[k][j] * matrix_B[j][i]
                res += str(new_elem) + ' '
            res += '\n'
        return res
    else:
        return 0
    
parser = argparse.ArgumentParser(description = "Matrix myltiply")
parser.add_argument('files', metavar= 'input and ouput files', type= str, nargs= 2)

files = parser.parse_args().files       
matrices = open(files[0], 'r').read().split("\n\n")
first_part = matrices[0].split('\n')
second_part = matrices[1].split('\n')
matrix_A = read_matrix(first_part)
matrix_B = read_matrix(second_part)

output = open(files[1], 'w')
res = matrix_mult(matrix_A, matrix_B)
if res == 0:
    output.writelines("Error: sizes of matrices are not correct")
else:
    output.writelines(res)