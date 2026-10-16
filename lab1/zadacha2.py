import math
import matplotlib.pyplot as plt

EPS = 0.001

def horner(coeffs, x):
    result = [coeffs[0]]
    for c in coeffs[1:]:
        result.append(result[-1] * x + c)
    return result

def upper_bound(coeffs):
    for x in range(1, 501):
        if all(v >= -1e-12 for v in horner(coeffs, x)):
            return x
    raise ValueError('Верхняя граница не найдена')

def lower_bound(coeffs):
    n = len(coeffs) - 1
    transformed = [c * (-1)**i for i, c in enumerate(coeffs)]

    return -upper_bound(transformed)

def find_intervals(f, a, b, step=0.1):
    result = []
    count = math.ceil((b - a) / step)
    for k in range(count):
        left = a + k * step
        right = min(b, left + step)
        if f(left) == 0 or f(left) * f(right) < 0:
            result.append((left, right))
    return result

def iteration_method(phi, x0, q, eps=EPS, max_iter=1000):
    history = [x0]
    for k in range(1, max_iter + 1):
        x_next = phi(history[-1])
        if not math.isfinite(x_next):
            raise ArithmeticError('Итерации перестали быть конечными')
        history.append(x_next)
        if q / (1 - q) * abs(history[-1] - history[-2]) < eps:
            return x_next, k, history
    raise RuntimeError('Не достигнута заданная точность')

def plot_graph(f, a, b, root, title):
    xs = [a + (b-a)*i/500 for i in range(501)]
    ys = [f(x) for x in xs]
    plt.figure(figsize=(8, 4.8))
    plt.plot(xs, ys, label='f(x)')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.plot(root, 0, 'ro', label=f'Корень = {root:.6f}')
    plt.axvline(root, color='red', linestyle='--', alpha=0.5)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

print('1) x^3 - 3x^2 + 9x + 3 = 0')
coeffs = [1, -3, 9, 3]
lo, hi = lower_bound(coeffs), upper_bound(coeffs)
print(f'Границы по Горнеру: [{lo}; {hi}]')
print('Строка Горнера для верхней границы:', horner(coeffs, hi))
transformed = [c * (-1)**i for i, c in enumerate(coeffs)]
print('Строка Горнера для нижней границы:', horner(transformed, -lo))
f1 = lambda x: x**3 - 3*x**2 + 9*x + 3
intervals1 = find_intervals(f1, lo, hi)
print('Интервалы отделения:', intervals1)

phi1 = lambda x: -3 / (x*x - 3*x + 9)

x1, n1, history1 = iteration_method(phi1, -0.3, q=0.2)
for i, x in enumerate(history1):
    print(f'  Итерация {i}: x = {x:.9f}')
print(f'Ответ: x = {x1:.3f}; итераций: {n1}; невязка = {abs(f1(x1)):.3g}\n')

print('2) (2-x)*e^x = 0')
f2 = lambda x: (2-x)*math.exp(x)
intervals2 = find_intervals(f2, 1, 3)
print('Интервалы отделения:', intervals2)

phi2 = lambda x: 2.0
x2, n2, history2 = iteration_method(phi2, 1.5, q=0.0)
for i, x in enumerate(history2):
    print(f'  Итерация {i}: x = {x:.9f}')
print(f'Ответ: x = {x2:.3f}; итераций: {n2}; невязка = {abs(f2(x2)):.3g}')

plot_graph(f1, -1, 0.5, x1, 'Задание 2.1: кубическое уравнение')
plot_graph(f2, 0, 3, x2, 'Задание 2.2: (2-x)e^x = 0')
plt.show()
