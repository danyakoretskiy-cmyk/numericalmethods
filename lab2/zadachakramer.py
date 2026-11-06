import copy
import time

DET_EPS = 1e-12
MAX_ITER = 1000
EPS = 1e-8


def cut_minor(matrix, column):
    temp = copy.deepcopy(matrix)
    temp.pop(0)
    for row in temp:
        row.pop(column)
    return temp


def determinant(matrix):
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    total = 0.0
    for j in range(len(matrix)):
        total += (-1) ** j * matrix[0][j] * determinant(cut_minor(matrix, j))
    return total


def cramer(A, b):
    d = determinant(A)
    if abs(d) < DET_EPS:
        raise ValueError('Определитель равен нулю: единственного решения нет')

    result = []
    for j in range(len(A)):
        changed = copy.deepcopy(A)
        for i in range(len(A)):
            changed[i][j] = b[i]
        result.append(determinant(changed) / d)
    return result


def residual(A, x, b):
    return max(abs(sum(A[i][j] * x[j] for j in range(len(A))) - b[i])
               for i in range(len(A)))


def seidel(A, b, eps):
    n = len(A)
    if any(abs(A[i][i]) < DET_EPS for i in range(n)):
        raise ValueError('Нулевой диагональный коэффициент')
    x = [b[i] / A[i][i] for i in range(n)]
    for k in range(1, MAX_ITER + 1):
        x_old = x.copy()
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - s) / A[i][i]
        diff = max(abs(x[i] - x_old[i]) for i in range(n))
        if diff < eps and residual(A, x, b) < eps:
            return x, k
    raise RuntimeError('Метод Зейделя не достиг заданной точности')


def is_diag_dominant(A):
    return all(abs(A[i][i]) > sum(abs(A[i][j]) for j in range(len(A)) if i != j)
               for i in range(len(A)))


def seidel_iteration_matrix(A):
    # B = -(D + L)^(-1) U. Столбцы B находим прямой подстановкой,
    # не используя готовые функции решения систем линейных уравнений.
    n = len(A)
    B = [[0.0] * n for _ in range(n)]
    for column in range(n):
        for i in range(n):
            upper = -A[i][column] if column > i else 0.0
            lower = sum(A[i][j] * B[j][column] for j in range(i))
            B[i][column] = (upper - lower) / A[i][i]
    return B


def multiply(M, N):
    size = len(M)
    return [[sum(M[i][k] * N[k][j] for k in range(size)) for j in range(size)]
            for i in range(size)]


def infinity_norm(M):
    return max(sum(abs(value) for value in row) for row in M)


def convergence_check(A):
    B = seidel_iteration_matrix(A)
    current = B
    for power in range(1, 21):
        norm = infinity_norm(current)
        if norm < 1:
            # ||B^power||_inf < 1 гарантирует rho(B) < 1.
            return True, power, norm
        current = multiply(current, B)
    return False, None, None  # отсутствие гарантии, НЕ доказательство расходимости


A = [[3, -0.7, 0.6, -2],
     [2, 5, -1, -3],
     [-1, 2, 4, 1],
     [2, 1, -1, 5]]
b = [11.5, 38.7, 12.2, 28.4]

if __name__ == '__main__':
    start = time.perf_counter()
    x_c = cramer(A, b)
    cramer_time = time.perf_counter() - start

    converges, power, norm = convergence_check(A)
    if not converges:
        print('Сходимость не доказана проверкой степеней матрицы')
    else:
        print(f'Сходимость доказана: ||B^{power}||_inf = {norm:.10f} < 1')

    start = time.perf_counter()
    x_s, iters = seidel(A, b, EPS)
    seidel_time = time.perf_counter() - start

    print('Лабораторная работа № 2, вариант 47')
    print('Определитель:', determinant(A))
    print('Строгое диагональное преобладание:', is_diag_dominant(A))
    print('Крамер:', x_c)
    print('Зейдель:', x_s)
    print('Невязка Крамера:', residual(A, x_c, b))
    print('Невязка Зейделя:', residual(A, x_s, b))
    print('Максимальное расхождение:', max(abs(a - b) for a, b in zip(x_c, x_s)))
    print('Итераций Зейделя:', iters)
    print('Время Крамера:', cramer_time, 'с')
    print('Время Зейделя:', seidel_time, 'с')
    if seidel_time > 0:
        print('Отношение времени:', cramer_time / seidel_time)
