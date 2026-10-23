import math
import matplotlib.pyplot

EPS = 0.001

def find_intervals(f, a, b, step=0.05):
    found = []
    n = math.ceil((b-a)/step)
    for i in range(n):
        x1 = a + i*step
        x2 = min(b, x1 + step)
        if f(x1)*f(x2) < 0:
            found.append((x1, x2))
    return found

def chord_method(f, a, b, eps=EPS, max_iter=1000):
    if f(a)*f(b) >= 0:
        raise ValueError('На концах интервала должна быть смена знака')
    history = []
    for i in range(1, max_iter+1):
        fa, fb = f(a), f(b)
        x = a - fa*(b-a)/(fb-fa)
        fx = f(x)
        history.append((a, b, x))
        if abs(fx) < eps * 0.1 or min(abs(x-a), abs(x-b)) < 1e-12:
            return x, i, history
        if fa*fx < 0:
            b = x
        else:
            a = x
    raise RuntimeError('Метод хорд не сошёлся')

def tangent_method(f, df, d2f, a, b, eps=EPS, max_iter=100):
    x = a if f(a)*d2f(a)>0 else b
    history = []
    for i in range(1, max_iter+1):
        derivative = df(x)
        if abs(derivative) < 1e-12:
            raise ZeroDivisionError('Производная близка к нулю')
        next_x = x - f(x)/derivative
        history.append((x, next_x))
        if abs(next_x-x) < eps:
            return next_x, i, history
        x = next_x
    raise RuntimeError('Метод Ньютона не сошёлся')

def combined_method(f, df, d2f, a, b, eps=EPS, max_iter=100):
    if f(a)*f(b) >= 0:
        raise ValueError('Нужен интервал со сменой знака')
    history = []
    for i in range(1, max_iter+1):
        fa, fb = f(a), f(b)
        chord = a - fa*(b-a)/(fb-fa)
        if fa*d2f(a)>0:
            tangent = a - fa/df(a)
            new_a, new_b = tangent, chord
            tan_from_a = True
        else:
            tangent = b - fb/df(b)
            new_a, new_b = chord, tangent
            tan_from_a = False
        history.append((a, b, chord, tangent, tan_from_a))
        a, b = sorted((new_a, new_b))
        if b-a < 2*eps:
            return (a+b)/2, i, history
    raise RuntimeError('Комбинированный метод не сошёлся')

def plot_method(f, root, left, right, history, method, title):
    pad = (right-left)*0.35
    xs = [left-pad + (right-left+2*pad)*i/500 for i in range(501)]
    matplotlib.pyplot.figure(figsize=(8.5, 5))
    matplotlib.pyplot.plot(xs, [f(x) for x in xs], linewidth=2, label='f(x)')
    matplotlib.pyplot.axhline(0, color='black', linewidth=0.8)
    for i, item in enumerate(history):
        if method == 'chord':
            a, b, x = item
            matplotlib.pyplot.plot([a,b],[f(a),f(b)],'--',color='orange',alpha=0.4)
        elif method == 'tangent':
            x0, x = item
            matplotlib.pyplot.plot([x0,x],[f(x0),0],color='green',alpha=0.5)
        else:
            a,b,chord,tangent,tan_from_a = item
            matplotlib.pyplot.plot([a,b],[f(a),f(b)],'--',color='orange',alpha=0.5)
            x0 = a if tan_from_a else b
            matplotlib.pyplot.plot([x0,tangent],[f(x0),0],color='green',alpha=0.5)
    matplotlib.pyplot.plot(root, 0, 'ro', label=f'Корень: {root:.6f}')
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.xlabel('x')
    matplotlib.pyplot.ylabel('f(x)')
    matplotlib.pyplot.grid(True, alpha=0.3)
    matplotlib.pyplot.legend()
    matplotlib.pyplot.tight_layout()

print('а) tan(0.5x + 0.1) = x^2 — метод хорд')
f_a = lambda x: math.tan(0.5*x+0.1)-x*x
print('Отделённые интервалы:', find_intervals(f_a, 0.5, 0.9))
a_a, b_a = 0.65, 0.7
root_a, count_a, history_a = chord_method(f_a, a_a, b_a)
for i, (_, _, x) in enumerate(history_a,1):
    print(f'  Итерация {i}: x = {x:.9f}')
print(f'Корень: {root_a:.3f}; итераций: {count_a}; |f(x)|={abs(f_a(root_a)):.5g}\n')

print('б) x^2+x-5=0 — метод касательных')
f_b = lambda x: x*x+x-5
df_b = lambda x: 2*x+1
d2f_b = lambda x: 2
print('Отделённые интервалы:', find_intervals(f_b, -3.5, 2.5))
a_b, b_b = 1.7, 1.8
root_b, count_b, history_b = tangent_method(f_b, df_b, d2f_b, a_b, b_b)
for i, (x0, x) in enumerate(history_b,1):
    print(f'  Итерация {i}: {x0:.9f} -> {x:.9f}')
print(f'Корень: {root_b:.3f}; итераций: {count_b}; |f(x)|={abs(f_b(root_b)):.5g}\n')

print('в) x^3+3x^2-2=0 — комбинированный метод')
f_c = lambda x: x**3+3*x*x-2
df_c = lambda x: 3*x*x+6*x
d2f_c = lambda x: 6*x+6
print('Отделённые интервалы:', find_intervals(f_c, -3, 1))
a_c, b_c = 0.7, 0.8
root_c, count_c, history_c = combined_method(f_c, df_c, d2f_c, a_c, b_c)
for i, (a,b,chord,tangent,_) in enumerate(history_c,1):
    print(f'  Итерация {i}: хорда = {chord:.9f}; касательная = {tangent:.9f}')
print(f'Корень: {root_c:.3f}; итераций: {count_c}; |f(x)|={abs(f_c(root_c)):.5g}')

plot_method(f_a, root_a, a_a, b_a, history_a, 'chord', 'а) Метод хорд')
plot_method(f_b, root_b, a_b, b_b, history_b, 'tangent', 'б) Метод касательных')
plot_method(f_c, root_c, a_c, b_c, history_c, 'combined', 'в) Комбинированный метод')
matplotlib.pyplot.show()
