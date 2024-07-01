import numpy as np

# 本文件用于处理作业一的优化问题，包含了对于 Proximal Gradient Method 以及 FISTA 的处理


class OptimizationSolver:
    def __init__(self, a_matrix, b_vec, _mu=1e-2):
        self.a_matrix = a_matrix
        self.b_vec = b_vec
        self.m, self.dim = a_matrix.shape
        self._lambda = 1 / (2 * self.m)
        self._mu = _mu                     # 文档中的量
        self.max_iter = 10000               # 最大循环数
        self.epi = 1e-6                     # 误差

    # 计算 l(x) 里面的光滑项
    def logistic_loss_smooth_part(self, x):
        loss = (1 / self.m) * np.sum(np.log(1 + np.exp(np.clip(-self.b_vec * (self.a_matrix @ x.T), -100, 100)))) + \
               self._lambda * np.linalg.norm(x, 2) ** 2
        return loss

    # 计算 l(x) 里面光滑项的导数
    def logistic_gradient_smooth_part(self, x):
        grad = ((-1 / self.m) * self.a_matrix.T @ (self.b_vec / (1 + np.exp(np.clip(self.b_vec * (self.a_matrix @ x.T), -100, 100)))) +
                2 * self._lambda * x)
        return grad

    # 计算 prox_l1
    def prox_l1(self, x, t):
        return np.sign(x) * np.maximum(np.abs(x) - t * self._mu, 0)

    # 执行 proximal gradient method backtracking
    def backtracking(self, x_pre, x_next, t0, grad, gamma):
        logistic_loss_smooth_val = self.logistic_loss_smooth_part(x_pre)
        while (self.logistic_loss_smooth_part(x_next) > logistic_loss_smooth_val +
               grad.T @ (x_next - x_pre) + 1 / (2 * t0) * np.linalg.norm(x_next - x_pre, 2) ** 2):
            t0 = t0 * gamma
            y = x_pre - t0 * grad
            prox = self.prox_l1(y, t0)
            x_next = prox
        return x_next, t0

    # 进行 proximal_gradient_method 的实现
    def proximal_gradient_method(self, gamma):
        t0 = 1
        print('执行近似点梯度法：')
        x = np.zeros(self.dim)

        iter_num = 0        # 迭代的次数
        log_scale = []      # 最优条件取log
        lx_func = []        # 函数 l(x) 的值
        for k in range(self.max_iter):
            t = t0
            iter_num = k + 1

            # 执行近似点梯度法，但是 t 尚未更新
            grad = self.logistic_gradient_smooth_part(x)
            y = x - t * grad
            prox = self.prox_l1(y, t)
            x_next = prox

            # 执行 backtracking
            [x_next, t] = self.backtracking(x, x_next, t, grad, gamma)

            # 画图准备
            opt_condition = np.linalg.norm(x_next - x) / t
            log_scale.append(np.log10(opt_condition))
            # 计算 l(x) 值
            lx_func_val = self.logistic_loss_smooth_part(x_next) + self._mu * np.linalg.norm(x_next)
            lx_func.append(lx_func_val)

            # 提前中止
            if opt_condition < self.epi:
                print('近似点梯度法执行结束')
                break

            x = x_next

        return x, iter_num, log_scale, lx_func

    # 执行 FISTA 算法
    def FISTA_method(self, gamma):
        t0 = 1
        print('执行 FISTA 算法：')
        x1 = np.zeros(self.dim)
        x2 = np.ones(self.dim) * 0.1

        iter_num = 0    # 迭代的次数
        log_scale = []  # 最优条件取log
        lx_func = []    # 函数 l(x) 的值
        for k in range(self.max_iter):
            t = t0
            iter_num = k + 1

            # 梯度法迭代
            y = x2 + (k + 1) / (k + 4) * (x2 - x1)
            grad = self.logistic_gradient_smooth_part(y)
            z = y - t * grad
            prox = self.prox_l1(z, t)
            x_next = prox

            # backtracking
            [x_next, t] = self.backtracking(y, x_next, t, grad, gamma)

            # 画图准备
            opt_condition = np.linalg.norm(x_next - x2) / t
            log_scale.append(np.log10(opt_condition))

            # 计算 l(x) 值
            lx_func_val = self.logistic_loss_smooth_part(x_next) + self._mu * np.linalg.norm(x_next)
            lx_func.append(lx_func_val)

            # 提前中止
            if opt_condition < self.epi:
                print('FISTA执行结束')
                break

            x1 = x2
            x2 = x_next

        return x2, iter_num, log_scale, lx_func
