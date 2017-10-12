
def golden(fun, bound, tot=1e-6):
    """
    一维黄金分割找极小值
    :param fun: 1元函数
    :param bound: [lb,ub]
    :param tot: 最小区间长度
    :return: [x0, minimal_val]
    """
    a, b = bound
    p = 0.382  # 黄金分割 1-p ~= 0.618
    a_next = None
    b_next = None
    cnt = 0
    while b-a > tot:
        cnt+=1
        if a_next is None:
            a1 = a+p*(b-a)
            f_a1 = fun(a1)
            a_next = (a1, f_a1)
        if b_next is None:
            b1 = a+(1-p) * (b-a)
            f_b1 = fun(b1)
            b_next = (b1, f_b1)
        if a_next[1] < b_next[1]:
            b = b_next[0]
            b_next = a_next
            a_next = None
        else:
            a = a_next[0]
            a_next = b_next
            b_next = None
    return [(b+a)/2, fun((b+a)/2),cnt]




if __name__ == '__main__':
    
    fun = lambda x: x**4 - 14*x**3 + 60*x**2 - 70*x
    ret = golden(fun, [0, 2])
    from scipy.optimize import minimize_scalar
    print('my ans\n', ret)
    res = minimize_scalar(fun,bounds=[0,2],method='brent')
    print('scipy\n',res)