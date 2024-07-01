from DataReader import *
from OptimizationSolver import *
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':

    # 读取a9a数据集
    filename = '.\\a9a.txt'
    a_matrix, b_vec = DataReader(filename)

    # 根据目的做不同的处理
    # 如果 flag=0，我们会比较 近似点梯度法，FISTA 的迭代速度
    # 如果 flag=1，我们会比较不同 mu 大小下的稀疏度与迭代步数
    # 如果 flag=2，我们画出所需要的图
    flag = 2

    if flag == 0:
        # 定义待优化函数的 solver，此时的 mu 为 1e-2
        solver = OptimizationSolver(a_matrix, b_vec)

        # 近似点梯度法在迭代步长收缩参数取不同值时的计算结果
        x_prox_1, prox_iter_nums_1 = solver.proximal_gradient_method(gamma=0.9)[:2]
        x_prox_2, prox_iter_nums_2 = solver.proximal_gradient_method(gamma=0.7)[:2]
        x_prox_3, prox_iter_nums_3 = solver.proximal_gradient_method(gamma=0.5)[:2]

        # FISTA
        x_FISTA_1, FISTA_iter_nums_1 = solver.FISTA_method(gamma=0.9)[:2]
        x_FISTA_2, FISTA_iter_nums_2 = solver.FISTA_method(gamma=0.7)[:2]
        x_FISTA_3, FISTA_iter_nums_3 = solver.FISTA_method(gamma=0.5)[:2]

        print(f"gamma=0.9时，近似点梯度法的迭代步数为{prox_iter_nums_1},FISTA的迭代次数为{FISTA_iter_nums_1}")
        print(f"gamma=0.7时，近似点梯度法的迭代步数为{prox_iter_nums_2},FISTA的迭代次数为{FISTA_iter_nums_2}")
        print(f"gamma=0.5时，近似点梯度法的迭代步数为{prox_iter_nums_3},FISTA的迭代次数为{FISTA_iter_nums_3}")
    elif flag == 1:
        # 定义待优化函数的不同 mu 情况下的 solver
        solver_mu_1 = OptimizationSolver(a_matrix, b_vec, _mu=1e-3)
        solver_mu_2 = OptimizationSolver(a_matrix, b_vec, _mu=1e-2)
        solver_mu_3 = OptimizationSolver(a_matrix, b_vec, _mu=0.01)
        solver_mu_4 = OptimizationSolver(a_matrix, b_vec, _mu=0.05)

        # 计算 近似点梯度法 下，迭代步长收缩参数 gamma=0.7 时，不同 mu 带来的不同结果
        x_prox_1, prox_iter_nums_1 = solver_mu_1.proximal_gradient_method(gamma=0.7)[:2]
        x_prox_2, prox_iter_nums_2 = solver_mu_2.proximal_gradient_method(gamma=0.7)[:2]
        x_prox_3, prox_iter_nums_3 = solver_mu_3.proximal_gradient_method(gamma=0.7)[:2]
        x_prox_4, prox_iter_nums_4 = solver_mu_4.proximal_gradient_method(gamma=0.7)[:2]

        # 稀疏度计算
        sparse_prox_mu1 = np.count_nonzero(x_prox_1 == 0) / a_matrix.shape[1]
        sparse_prox_mu2 = np.count_nonzero(x_prox_2 == 0) / a_matrix.shape[1]
        sparse_prox_mu3 = np.count_nonzero(x_prox_3 == 0) / a_matrix.shape[1]
        sparse_prox_mu4 = np.count_nonzero(x_prox_4 == 0) / a_matrix.shape[1]

        # 画出表格
        sparse_prox = pd.DataFrame({
            'mu': [1e-3, 1e-2, 0.01, 0.05],
            '稀疏度': [sparse_prox_mu1, sparse_prox_mu2, sparse_prox_mu3, sparse_prox_mu4],
            '迭代步数': [prox_iter_nums_1, prox_iter_nums_2, prox_iter_nums_3, prox_iter_nums_4]
        })

        print(f"近似点梯度法调整 mu 值得到的表格:\n{sparse_prox}")

        # 计算 FISTA 方法下，迭代步长收缩参数 gamma=0.7 时，不同 mu 带来的不同结果
        x_FISTA_1, FISTA_iter_nums_1 = solver_mu_1.FISTA_method(gamma=0.7)[:2]
        x_FISTA_2, FISTA_iter_nums_2 = solver_mu_2.FISTA_method(gamma=0.7)[:2]
        x_FISTA_3, FISTA_iter_nums_3 = solver_mu_3.FISTA_method(gamma=0.7)[:2]
        x_FISTA_4, FISTA_iter_nums_4 = solver_mu_4.FISTA_method(gamma=0.7)[:2]

        # 稀疏度计算
        sparse_FISTA_mu1 = np.count_nonzero(x_FISTA_1 == 0) / a_matrix.shape[1]
        sparse_FISTA_mu2 = np.count_nonzero(x_FISTA_2 == 0) / a_matrix.shape[1]
        sparse_FISTA_mu3 = np.count_nonzero(x_FISTA_3 == 0) / a_matrix.shape[1]
        sparse_FISTA_mu4 = np.count_nonzero(x_FISTA_4 == 0) / a_matrix.shape[1]

        # 画出表格
        sparse_FISTA = pd.DataFrame({
            'mu': [1e-3, 1e-2, 0.01, 0.05],
            '稀疏度': [sparse_FISTA_mu1, sparse_FISTA_mu2, sparse_FISTA_mu3, sparse_FISTA_mu4],
            '迭代步数': [FISTA_iter_nums_1, FISTA_iter_nums_2, FISTA_iter_nums_3, FISTA_iter_nums_4]
        })

        print(f"FISTA 方法调整 mu 值得到的表格:\n{sparse_FISTA}")
    elif flag == 2:
        # 当 mu=1e-2, gamma=0.7 时的作图展示
        solver = OptimizationSolver(a_matrix, b_vec)

        x_prox, iter_num_prox, log_scale_prox, lx_func_prox = solver.proximal_gradient_method(gamma=0.7)
        x_FISTA, iter_num_FISTA, log_scale_FISTA, lx_func_FISTA = solver.FISTA_method(gamma=0.7)

        # 最优条件与迭代步数的关系
        plt.figure(1)
        step_range_prox = list(range(1, iter_num_prox+1, 1))
        step_range_FISTA = list(range(1, iter_num_FISTA+1, 1))
        plt.plot(step_range_prox, log_scale_prox, color='green', linewidth=1.0,
                 linestyle='--', label='proximal gradient method')
        plt.plot(step_range_FISTA, log_scale_FISTA, color='blue',
                 linewidth=1.0, linestyle='--', label='FISTA')
        plt.title('Relationship between optimal condition and iteration steps')
        plt.xlim(0, 1200)
        plt.ylim(-6, 6)
        plt.xlabel('iteration steps')
        plt.ylabel('log scale')
        x_tricks = np.linspace(0, 1200, 100)
        plt.xticks = (x_tricks)
        y_tricks = np.linspace(-6, 6, 2)
        plt.yticks = (y_tricks)
        plt.legend(loc='best')

        # 函数误差与迭代点关系
        # 计算函数误差
        x_prox_l_val = solver.logistic_loss_smooth_part(x_prox) + 1e-2 * np.linalg.norm(x_prox)
        print(f"the result l(x) of proximal gradient method is {x_prox_l_val}")
        error_prox = [x - x_prox_l_val for x in lx_func_prox]

        x_FISTA_l_val = solver.logistic_loss_smooth_part(x_FISTA) + 1e-2 * np.linalg.norm(x_FISTA)
        print(f"the result l(x) of FISTA is {x_FISTA_l_val}")
        error_FISTA = [x - x_FISTA_l_val for x in lx_func_FISTA]

        plt.figure(2)
        plt.plot(step_range_prox, error_prox, color='green', linewidth=1.0,
                 linestyle='--', label='proximal gradient')
        plt.plot(step_range_FISTA, error_FISTA, color='blue',
                 linewidth=1.0, linestyle='--', label='FISTA')
        plt.title('Relationship between function error and iteration steps')
        plt.xlim(0, 1200)
        plt.ylim(1e-3, 1)
        plt.xlabel('iteration steps')
        plt.ylabel('function error')
        x_tricks = np.linspace(0, 1200, 100)
        plt.xticks = (x_tricks)
        plt.yticks = ([1e-3, 1e-2, 1e-1, 1])
        plt.yscale('log')
        plt.legend(loc='best')

        plt.show()
