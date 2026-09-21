def cal_gauss(A, x, b):
    n = len(A)
    A_ = [row[:] for row in A]
    b_ = b[:]
    
    for k in range(n - 1):
        max_index = k
        for i in range(k, n):
            if abs(A_[i][k]) > abs(A_[max_index][k]):
                max_index = i
        if abs(A_[max_index][k]) < 1e-20:
            raise ValueError("Matrix is singular")
        for j in range(k, n):
            A_[k][j], A_[max_index][j] = A_[max_index][j], A_[k][j]
        b_[k], b_[max_index] = b_[max_index], b_[k]
        
        for i in range(k + 1, n):
            if A_[i][k] != 0.0:
                lam = A_[i][k] / A_[k][k]
                for j in range(k + 1, n):
                    A_[i][j] -= lam * A_[k][j]
                b_[i] -= lam * b_[k] 
                
    for i in range(n - 1, -1, -1):
        x[i] = b_[i]
        for j in range(i + 1, n):
            x[i] -= A_[i][j] * x[j]
        x[i] /= A_[i][i]
    
    return x

def cal_cholesky(A, x, b):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    D = [0.0] * n

    for i in range(n):
        L[i][i] = 1.0

    # ---------- 分解 ----------
    D[0] = A[0][0]
    if D[0] <= 0:
        raise ValueError(f"Matrix is not positive definite (d_0 = {D[0]})")

    for i in range(1, n):
        t = [0.0] * i
        for j in range(i):
            s = 0.0
            for k in range(j):
                s += t[k] * L[j][k]
            t[j] = A[i][j] - s
            L[i][j] = t[j] / D[j]
        s = 0.0
        for k in range(i):
            s += t[k] * L[i][k]
        D[i] = A[i][i] - s
        if D[i] <= 0:
            print(f"Matrix is not positive definite (d_{i} = {D[i]})")

    y = [0.0] * n
    for i in range(n):
        y[i] = b[i]
        for k in range(i):
            y[i] -= L[i][k] * y[k]


    for i in range(n - 1, -1, -1):
        x[i] = y[i] / D[i]
        for k in range(i + 1, n):
            x[i] -= L[k][i] * x[k]

    return x

def cal_tikhonov(A, b, alpha=0.01):
    n = len(A)

    # 构造 A^T A + alpha * I
    ATA = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[k][i] * A[k][j]
            ATA[i][j] = s
        ATA[i][i] += alpha

    # 构造 A^T b
    ATb = [0.0] * n
    for i in range(n):
        s = 0.0
        for k in range(n):
            s += A[k][i] * b[k]
        ATb[i] = s

    # 解 (A^T A + alpha I)x = A^T b
    x = [0.0] * n
    return cal_cholesky(ATA, x, ATb)

def main():
    for n in [10, 11, 12, 13, 14, 15]:
        H = [[1 / (i + j + 1) for j in range(n)] for i in range(n)]
        x = [1] * n       
        b = [0] * n
        for i in range(n):
            for j in range(n):
                b[i] += H[i][j] * x[j]    
        try:
            print(f"Solution for n={n} using Gauss elimination: {cal_gauss(H, x, b)}")
        except ValueError as e:
            print(f"Error in Gauss elimination for n={n}: {e}")
        try:
            print(f"Solution for n={n} using Cholesky decomposition: {cal_cholesky(H, x, b)}")
        except ValueError as e:
            print(f"Error in Cholesky decomposition for n={n}: {e}")
        try:
            print(f"Solution for n={n} using Tikhonov regularization: {cal_tikhonov(H, b, alpha=1e-12)}")
        except ValueError as e:
            print(f"Error in Tikhonov regularization for n={n}: {e}")
if __name__ == "__main__":
    main()