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
        if _ % (len(x0)+5) ==0:
            d0 = -g_k
    return OptimizeResult({'x': x0, 'fun': fun(*((x0,) + args)), 'jac': grad(*((x0,) + g_args)), 'nit': max_iter - _})


if __name__ == '__main__':
    from scipy.optimize import rosen, rosen_der, minimize

    # res = fast_gradient(rosen, rosen_der, [-2, 2], tol=1e-8, max_iter=100000)
    # print('my opt\n', res)
    # res = cg_gradient(rosen, rosen_der, [-2, 2])
    # print('cg res\n')
    # print(res)
    # print('scipy.opt.minimize\n')
    # res = minimize(rosen, x0=[-2, 2], jac=rosen_der)
    # print(res)

    f