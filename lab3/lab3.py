"""Численные методы. Лабораторная работа №3, вариант 47.
Только стандартная библиотека Python. Точность по невязке: 0.001.
"""
import math
import time

EPS = 1e-3
MAX_ITER = 100000


def norm_inf(v):
    return max(abs(z) for z in v)


def newton_functions(x, y):
    return (math.cos(x + 0.5) - y - 2, math.sin(y) - 2*x - 1)


def newton_jacobian(x, y):
    return (-math.sin(x + 0.5), -1, -2, math.cos(y))


def solve_2x2(a, b, c, d, e, f):
    det = a*d - b*c
    if abs(det) < 1e-12:
        raise ValueError('Якобиан вырожден: нельзя решить систему поправок')
    return ((e*d - b*f)/det, (a*f - e*c)/det)


def modified_newton(x0, y0, eps=EPS):
    """Якобиан вычисляется один раз в начальной точке."""
    a,b,c,d = newton_jacobian(x0, y0)
    x,y=x0,y0
    for k in range(MAX_ITER+1):
        f1,f2=newton_functions(x,y)
        if norm_inf((f1,f2)) < eps:
            return (x,y,k,norm_inf((f1,f2)))
        if k == MAX_ITER:
            break
        dx,dy=solve_2x2(a,b,c,d,-f1,-f2)
        x+=dx
        y+=dy
        if not (math.isfinite(x) and math.isfinite(y)):
            break
    raise RuntimeError('Модифицированный Ньютон не сошёлся')


def gradient_functions(x,y):
    z=x*y
    if abs(math.cos(z)) < 1e-10:
        raise ValueError('Функция tan(x*y) не определена')
    return (math.tan(z)-x*x, 7*x*x+2*y*y-1)


def objective(x,y):
    f1,f2=gradient_functions(x,y)
    return 0.5*(f1*f1+f2*f2)


def gradient(x,y):
    f1,f2=gradient_functions(x,y)
    sec2=1/(math.cos(x*y)**2)
    df1dx=y*sec2-2*x
    df1dy=x*sec2
    df2dx=14*x
    df2dy=4*y
    return (f1*df1dx+f2*df2dx, f1*df1dy+f2*df2dy)


def gradient_descent(x0,y0,eps=EPS):
    """Минимизирует 0.5*(f1^2+f2^2), шаг — обратный поиск Армихо."""
    x,y=x0,y0
    for k in range(MAX_ITER+1):
        f1,f2=gradient_functions(x,y)
        resid=norm_inf((f1,f2))
        if resid < eps:
            return (x,y,k,resid)
        if k == MAX_ITER:
            break
        gx,gy=gradient(x,y)
        g2=gx*gx+gy*gy
        if g2 < 1e-24:
            raise RuntimeError('Градиент слишком мал при ненулевой невязке')
        old=0.5*(f1*f1+f2*f2)
        step=1.0
        for _ in range(70):
            nx,ny=x-step*gx,y-step*gy
            try:
                new=objective(nx,ny)
            except (ValueError, OverflowError):
                new=math.inf
            if math.isfinite(new) and new <= old - 1e-4*step*g2:
                x,y=nx,ny
                break
            step*=0.5
        else:
            raise RuntimeError('Не удалось подобрать убывающий шаг')
    raise RuntimeError('Градиентный метод не сошёлся')


def print_result(name, method, guess):
    t0=time.perf_counter()
    try:
        x,y,steps,resid=method(*guess)
    except (ValueError,RuntimeError,OverflowError) as exc:
        print(f'{name}: ошибка: {exc}')
        return
    elapsed=time.perf_counter()-t0
    print(f'{name}; начальное приближение {guess}')
    print(f'  x = {x:.9f}; y = {y:.9f}')
    print(f'  итераций = {steps}; максимальная невязка = {resid:.3e}')
    print(f'  время = {elapsed:.6f} секунд')


if __name__ == '__main__':
    print('Лабораторная №3, вариант 47. Требуемая точность: 0.001')
    print('\nЗадание Б. Модифицированный метод Ньютона')
    print_result('Ньютон',modified_newton,(-1.0,-1.0))
    print('\nЗадание Г. Градиентный метод')
    for start in ((0.3,0.4),(0.0,0.6),(0.0,-0.6)):
        print_result('Градиент',gradient_descent,start)
