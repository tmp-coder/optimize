import numpy as np
from numpy.linalg import norm
from scipy.optimize import minimize_scalar, OptimizeResult


def is_stop(next_val, current, tol):
    """
    停机准则梯度方法
    :param next_val:
    :param current:
    :return: bool
    """

    return norm(next_val - current, 2) / max(1, norm(current, 2)) < tol


def fast_gradient(fun, grad, x0, tol=1e-7, max_iter=500):
    phi = lambda alpha, x: fun(x - alpha * np.array(grad(x)))  # 最速降参数
    iters = max_iter
    while iters > 0:
        iters -= 1
        res = minimize_scalar(phi, method='brent', args=x0, tol=1e-5)
        x_next = x0 - res.x * np.array(grad(x0))
        if is_stop(x_next, x0, tol):
            break
        x0 = x_next
    return OptimizeResult({'x': x0, 'fun': fun(x0), 'jac': grad(x0), 'nit': max_iter - iters})


def cg_gradient(fun, grad, x0, args=(), g_args=(), tol=1e-8, max_iter=5000):
    alpha = lambda a, x_k, d: fun(*((x_k + a * d,) + args))
    g0 = grad(*((x0,) + g_args))
    d0 = -g0
    for _ in range(max_iter):
        a_k = minimize_scalar(alpha, bounds=(0, 100), args=(x0, d0), tol=1e-4)
        x0 = x0 + a_k.x * d0
        g_k = grad(*((x0,) + g_args))
        if is_stop(g_k, np.zeros(g_k.shape), tol):
            break
        beta = np.sum(g_k ** 2) / np.sum(g0 ** 2)  # Fletcher-Reeves 公式
        g0 = g_k
        d0 = -g_k + beta * d0
        if _ % (len(x0) + 5) == 0:
            d0 = -g_k
    return OptimizeResult({'x': x0, 'fun': fun(*((x0,) + args)), 'jac': grad(*((x0,) + g_args)), 'nit': max_iter - _})


def dfp(fun, grad, x0, args=(), g_args=(), tol=1e-8, max_iter=5000):
    """

    :param fun: function ，目标函数
    :param grad:function ，目标函数梯度
    :param x0: list， 初始向量
    :param args: tuple， fun，其余参数
    :param g_args: tuple， grad其余参数
    :param tol: float，精度
    :param max_iter: int， 最大迭代次数
    :return: OptimizeResult, 最优解
    """
    h0 = np.eye(len(x0))
    g_0 = grad(*((x0,) + g_args))
    alpha = lambda a, x, d: fun(*((x + a * d,) + args))
    for i in range(max_iter):
        if is_stop(g_0, np.zeros(g_0.shape), tol):
            break
        d = -h0.dot(g_0)
        alp = minimize_scalar(alpha, bounds=(0, 10000), args=(x0, d), method='brent', tol=1e-4)
        alp = alp.x
        x_next = x0 + alp * d
        delta_x = (alp * d).reshape((len(x0), 1))
        g_next = grad(*((x_next,) + g_args))
        delta_g = g_next - g_0
        delta_g = delta_g.reshape((delta_x.shape))
        tmp = h0.dot(delta_g)
        h0 = h0 + delta_x.dot(delta_x.T) / (delta_x.T.dot(delta_g)) - tmp.dot(tmp.T) / (delta_g.T.dot(tmp))
        x0 = x_next
        g_0 = g_next
    return OptimizeResult({'nit': i, 'x': x0, 'jac': g_0, 'fun': fun(*((x0,) + args))})


def bfgs(fun, grad, x0, args=(), g_args=(), tol=1e-8, max_iter=5000):
    """

    :param fun: function ，目标函数
    :param grad:function ，目标函数梯度
    :param x0: list， 初始向量
    :param args: tuple， fun，其余参数
    :param g_args: tuple， grad其余参数
    :param tol: float，精度
    :param max_iter: int， 最大迭代次数
    :return: OptimizeResult, 最优解
    """
    h0 = np.eye(len(x0))
    g_0 = grad(*((x0,) + g_args))
    alpha = lambda a, x, d: fun(*((x + a * d,) + args))
    for i in range(max_iter):
        if is_stop(g_0, np.zeros(g_0.shape), tol):
            break
        d = -h0.dot(g_0)
        alp = minimize_scalar(alpha, bounds=(0, 10000), args=(x0, d), method='brent', tol=1e-4)
        alp = alp.x
        x_next = x0 + alp * d
        delta_x = (alp * d).reshape((len(x0), 1))
        g_next = grad(*((x_next,) + g_args))
        delta_g = g_next - g_0
        delta_g = delta_g.reshape((delta_x.shape))
        tmp = h0.dot(delta_g).dot(delta_x.T)
        tmp1 = (1+delta_g.T.dot(h0).dot(delta_g)/(delta_g.T.dot(delta_x)))*delta_x.dot(delta_x.T)/(delta_x.T.dot(delta_g))
        tmp2 = (tmp + tmp.T)/(delta_g.T.dot(delta_x))
        h0 = h0 + tmp1 - tmp2
        x0 = x_next
        g_0 = g_next
    return OptimizeResult({'nit': i, 'x': x0, 'jac': g_0, 'fun': fun(*((x0,) + args))})

