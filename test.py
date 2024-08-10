import torch
def rms_norm():
    # 创建一个3维矩阵
    matrix = torch.randn(2, 3, 2, 2, dtype=torch.float32)
    print("input = ")
    print(matrix)

    w = torch.tensor([1.0, 2.0], dtype=torch.float32)  # 示例权重
    epsilon = 1e-6       # 防止除零的小常数
    last_dim_size = matrix.size(-1)

    squares = torch.sum(matrix ** 2 , dim=-1, keepdim=True) / last_dim_size + epsilon
    print("squares = ")
    print(squares)

    norm = torch.sqrt(squares)
    print("norm = ")
    print(norm)

    y = (w * matrix) / norm
    print("result = ")
    print(y)


def matmul_trans(alpha, beta, m, k, n):

    # 创建矩阵 A 和 B，形状相同
    A = torch.randn(m, k, dtype=torch.float32)
    B = torch.randn(n, k, dtype=torch.float32)

    C = torch.randn(m, n, dtype=torch.float32)
    print("C = ")
    print(C)
    ABT = torch.matmul(A, B.T)

    C = alpha * ABT + beta * C
    print("A = ")
    print(A)
    print("B = ")
    print(B)
    print("C = ")
    print(C)


matmul_trans(1.41, 2.43, 2, 3, 4)