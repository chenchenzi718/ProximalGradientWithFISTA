from scipy.sparse import lil_matrix
import numpy as np

# 本文件用于读取文件 a9a 中的数据信息


def DataReader(filename):

    b_vec = []                  # 记录所有的 b 成为一个向量，大小为 32561*1
    a_matrix_value = []         # 之后记录所有的 a 成为一个矩阵，大小为 32561*123
    row_indices = []            # 记录稀疏矩阵的 row_index
    col_indices = []            # 记录稀疏矩阵的 col_index

    with open(filename, 'r') as file:
        for line in file:

            # 取出第 i 行内所有数据
            components = line.strip().split()

            # 取出这一行的第一位即 bi
            bi = int(components[0])
            b_vec.append(bi)

            # 取出 ai 信息，按照 col:value 的形式取出
            for item in components[1:]:
                col_idx, value = item.split(':')
                row_indices.append(len(b_vec) - 1)
                col_indices.append(int(col_idx) - 1)
                a_matrix_value.append(float(value))

    # m 就代表了作业文档里的 ai,bi 个数， dim 表示 x 的维数
    m = len(b_vec)
    dim = max(col_indices) + 1 if col_indices else 0

    # 创建稀疏矩阵 a，标签矩阵 b
    a_matrix = lil_matrix((m, dim))
    a_matrix[row_indices, col_indices] = a_matrix_value
    b_vec = np.array(b_vec)

    # 供测试使用，减少数据量
    # a_matrix = a_matrix[:500, :]
    # b_vec = b_vec[:500]

    return a_matrix, b_vec
