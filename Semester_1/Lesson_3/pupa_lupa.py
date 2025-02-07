class Worker:
    def __init__(self):
        self._salary = 0

    def take_salary(self, n):
        self._salary += n
    
    @property
    def salary(self):
        return self._salary
    
    def _read_matrix(self, filename):
        text = open(filename, 'r').read().split('\n')
        matrix = []
        i = 0
        for str in text:
            matrix.append([])
            for num in str.split():
                matrix[i].append(int(num))
            i += 1
        return matrix

class Pupa(Worker):
    def __init__(self):
        super().__init__()

    def do_work(self, matrix_file1, matrix_file2):
        mat_A = self._read_matrix(matrix_file1)
        mat_B = self._read_matrix(matrix_file2)

        lines_A = len(mat_A)
        collums_A = len(mat_A[0])
        lines_B = len(mat_B)
        collums_B = len(mat_B[0])
        if lines_A != lines_B or collums_A != collums_B:
            return []
        
        for i in range(lines_A):
            for j in range(collums_A):
                print(mat_A[i][j] + mat_B[i][j], ' ', sep='', end='')
            print()
        print()       

class Lupa(Worker):
    def __init__(self):
        super().__init__()

    def do_work(self, matrix_file1, matrix_file2):
        mat_A = self._read_matrix(matrix_file1)
        mat_B = self._read_matrix(matrix_file2)

        lines_A = len(mat_A)
        collums_A = len(mat_A[0])
        lines_B = len(mat_B)
        collums_B = len(mat_B[0])
        if lines_A != lines_B or collums_A != collums_B:
            raise Exception("Matrices sizes must be same.")
        
        for i in range(lines_A):
            for j in range(collums_A):
                print(mat_A[i][j] - mat_B[i][j], ' ', sep= '', end='')
            print()
        print()   

class Accountant:
    def __init__(self, salary=100):
        self.__salary = salary

    def give_salary(self, worker):
        worker.take_salary(self.__salary)

    def change_salary(self, new_salary):
        self.__salary = new_salary

def main():
    boss = Accountant()
    pupa = Pupa()
    lupa = Lupa()
    boss.give_salary(pupa)
    boss.give_salary(lupa)
    print(pupa.salary, lupa.salary, '\n')
    pupa.do_work('file1.txt', 'file2.txt')
    lupa.do_work('file1.txt', 'file2.txt')
main()