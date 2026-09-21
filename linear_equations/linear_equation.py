import math
import time


def cal_gauss(A, b):
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

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = b_[i]
        for j in range(i + 1, n):
            x[i] -= A_[i][j] * x[j]
        x[i] /= A_[i][i]

    return x


def cal_cholesky(A, b):
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

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = y[i] / D[i]
        for k in range(i + 1, n):
            x[i] -= L[k][i] * x[k]

    return x


def cal_tikhonov(A, b, alpha=0.01, method='cholesky'):
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
    if method == 'gauss':
        return cal_gauss(ATA, ATb)
    elif method == 'cholesky':
        return cal_cholesky(ATA, ATb)
    else:
        raise ValueError(f"Unknown method: {method}")


def cal_conjugate_gradient(A, b, x0=None, tol=1e-10, max_iter=1000):
    n = len(A)
    if x0 is None:
        x = [0.0] * n
    else:
        x = x0[:]

    r = [b[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
    p = r[:]
    rsold = sum(r[i] ** 2 for i in range(n))

    for _ in range(max_iter):
        Ap = [sum(A[i][j] * p[j] for j in range(n)) for i in range(n)]
        alpha = rsold / sum(p[i] * Ap[i] for i in range(n))

        for i in range(n):
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]

        rsnew = sum(r[i] ** 2 for i in range(n))
        if rsnew < tol ** 2:
            break

        beta = rsnew / rsold
        p = [r[i] + beta * p[i] for i in range(n)]
        rsold = rsnew

    return x


def cal_GMRES(A, b, x0=None, tol=1e-10, max_iter=None, restart=None):
    """
    GMRES(m) 求解 A x = b。
    A, b 用 Python 列表表示。
    restart=None 表示不重启。
    """
    n = len(A)
    if max_iter is None:
        max_iter = n
    if restart is None:
        restart = max_iter

    x = [0.0] * n if x0 is None else x0[:]

    total = 0
    while total < max_iter:
        # ---------- 当前残差 ----------
        r = [b[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
        beta = math.sqrt(sum(ri * ri for ri in r))
        if beta < tol:
            return x

        m = min(restart, max_iter - total)

        # ---------- 空间分配 ----------
        V = [[r[i] / beta for i in range(n)]]          # V[0..m]
        H = [[0.0] * m for _ in range(m + 1)]          # H[0..m][0..m-1]
        cs = [0.0] * m
        sn = [0.0] * m
        g = [0.0] * (m + 1)
        g[0] = beta

        niter = 0
        for k in range(m):
            niter = k + 1

            # ---------- Arnoldi 第 k 步 ----------
            w = [sum(A[i][j] * V[k][j] for j in range(n)) for i in range(n)]
            for j in range(k + 1):
                H[j][k] = sum(w[i] * V[j][i] for i in range(n))
                w = [w[i] - H[j][k] * V[j][i] for i in range(n)]
            hkp1 = math.sqrt(sum(wi * wi for wi in w))
            H[k + 1][k] = hkp1

            # ---------- 用已有 Givens 旋转作用到第 k 列 ----------
            for i in range(k):
                temp = cs[i] * H[i][k] + sn[i] * H[i + 1][k]
                H[i + 1][k] = -sn[i] * H[i][k] + cs[i] * H[i + 1][k]
                H[i][k] = temp

            # ---------- 构造新 Givens 旋转，消去 H[k+1][k] ----------
            denom = math.hypot(H[k][k], H[k + 1][k])
            if denom < 1e-300:
                cs[k], sn[k] = 1.0, 0.0
            else:
                cs[k] = H[k][k] / denom
                sn[k] = H[k + 1][k] / denom
                H[k][k] = denom
            H[k + 1][k] = 0.0

            # ---------- 同一旋转作用到右端项 g ----------
            g[k + 1] = -sn[k] * g[k]
            g[k]     =  cs[k] * g[k]

            # ---------- 用 |g[k+1]| 判断收敛 ----------
            if abs(g[k + 1]) < tol:
                break

            # ---------- lucky breakdown ----------
            if hkp1 < 1e-14:
                break

            V.append([w[i] / hkp1 for i in range(n)])

        # ---------- 回代：R y = g[0:niter] ----------
        y = [0.0] * niter
        for i in range(niter - 1, -1, -1):
            s = g[i]
            for j in range(i + 1, niter):
                s -= H[i][j] * y[j]
            y[i] = s / H[i][i] if abs(H[i][i]) > 1e-300 else 0.0

        # ---------- 更新解 x = x + V y ----------
        for i in range(niter):
            yi = y[i]
            if yi != 0.0:
                Vi = V[i]
                for j in range(n):
                    x[j] += yi * Vi[j]

        total += niter

    return x


def evaluate_error(x, x_true, A=None, b=None):
    """
    计算 x 相对真实解 x_true 的误差以及残差。
    返回一个 dict。
    """
    n = len(x)
    diff = [x[i] - x_true[i] for i in range(n)]

    # 无穷范数
    err_inf = max(abs(d) for d in diff)
    # 2-范数
    err_2 = math.sqrt(sum(d * d for d in diff))
    # 相对 2-范数误差
    norm_true = math.sqrt(sum(v * v for v in x_true))
    rel_err = err_2 / norm_true if norm_true > 0 else float('inf')

    # 残差
    res_inf = None
    if A is not None and b is not None:
        r = [b[i] - sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
        res_inf = max(abs(ri) for ri in r)

    return {
        'err_inf': err_inf,
        'err_2':   err_2,
        'rel_err': rel_err,
        'res_inf': res_inf,
    }


def format_err(e):
    return (f"err_inf={e['err_inf']:.3e}, "
            f"err_2={e['err_2']:.3e}, "
            f"rel_err={e['rel_err']:.3e}, "
            f"res_inf={e['res_inf']:.3e}"
            if e['res_inf'] is not None else
            f"err_inf={e['err_inf']:.3e}, "
            f"err_2={e['err_2']:.3e}, "
            f"rel_err={e['rel_err']:.3e}")


# ==================== 主程序 ====================
def main():
    # 结果文件
    out = open("linear_equation_results.txt", "w", encoding="utf-8")

    def log(s=""):
        print(s)
        out.write(s + "\n")

    log("=" * 90)
    log("线性方程组数值方法测试 —— Hilbert 矩阵 H_n, 真解 x* = (1,...,1)^T")
    log("=" * 90)

    # 全局汇总表：Tikhonov 正则化分别用 Gauss 和 Cholesky 统计
    summary = {name: {'time': [], 'err_inf': []}
               for name in ['Gauss', 'Cholesky', 'Gauss+Reg',
                            'Cholesky+Reg', 'CG', 'GMRES']}

    for n in [10, 11, 12, 13, 14, 15]:
        log("")
        log("-" * 90)
        log(f"n = {n}")
        log("-" * 90)

        # Hilbert 矩阵与右端项
        H = [[1.0 / (i + j + 1) for j in range(n)] for i in range(n)]
        x_true = [1.0] * n
        b = [sum(H[i][j] * x_true[j] for j in range(n)) for i in range(n)]

        # 条件数估计（粗略）
        log(f"  Hilbert 矩阵 H_{n} 的条件数约为 1e{int(1.5 * n)} 量级（粗略）")

        # --- 1. Gauss 消元 ---
        t0 = time.perf_counter()
        try:
            x_gauss = cal_gauss(H, b)
            t1 = time.perf_counter()
            e = evaluate_error(x_gauss, x_true, H, b)
            log(f"[Gauss]     time={t1-t0:.6f}s  {format_err(e)}")
            log(f"            x[:5] = {x_gauss[:5]}")
            summary['Gauss']['time'].append(t1 - t0)
            summary['Gauss']['err_inf'].append(e['err_inf'])
        except Exception as ex:
            log(f"[Gauss]     失败: {ex}")
            summary['Gauss']['time'].append(None)
            summary['Gauss']['err_inf'].append(None)

        # --- 2. Cholesky 分解 ---
        t0 = time.perf_counter()
        try:
            x_chol = cal_cholesky(H, b)
            t1 = time.perf_counter()
            e = evaluate_error(x_chol, x_true, H, b)
            log(f"[Cholesky]  time={t1-t0:.6f}s  {format_err(e)}")
            log(f"            x[:5] = {x_chol[:5]}")
            summary['Cholesky']['time'].append(t1 - t0)
            summary['Cholesky']['err_inf'].append(e['err_inf'])
        except Exception as ex:
            log(f"[Cholesky]  失败: {ex}")
            summary['Cholesky']['time'].append(None)
            summary['Cholesky']['err_inf'].append(None)

        # --- 3. Tikhonov 正则化 + Gauss ---
        alpha = 1e-12
        t0 = time.perf_counter()
        try:
            x_reg_gauss = cal_tikhonov(H, b, alpha=alpha, method='gauss')
            t1 = time.perf_counter()
            e = evaluate_error(x_reg_gauss, x_true, H, b)
            log(f"[Gauss+Reg] time={t1-t0:.6f}s  {format_err(e)}")
            log(f"            x[:5] = {x_reg_gauss[:5]}")
            summary['Gauss+Reg']['time'].append(t1 - t0)
            summary['Gauss+Reg']['err_inf'].append(e['err_inf'])
        except Exception as ex:
            log(f"[Gauss+Reg] 失败: {ex}")
            summary['Gauss+Reg']['time'].append(None)
            summary['Gauss+Reg']['err_inf'].append(None)

        # --- 4. Tikhonov 正则化 + Cholesky ---
        t0 = time.perf_counter()
        try:
            x_reg_chol = cal_tikhonov(H, b, alpha=alpha, method='cholesky')
            t1 = time.perf_counter()
            e = evaluate_error(x_reg_chol, x_true, H, b)
            log(f"[Cholesky+Reg] time={t1-t0:.6f}s  {format_err(e)}")
            log(f"            x[:5] = {x_reg_chol[:5]}")
            summary['Cholesky+Reg']['time'].append(t1 - t0)
            summary['Cholesky+Reg']['err_inf'].append(e['err_inf'])
        except Exception as ex:
            log(f"[Cholesky+Reg] 失败: {ex}")
            summary['Cholesky+Reg']['time'].append(None)
            summary['Cholesky+Reg']['err_inf'].append(None)

        # --- 5. 共轭梯度法 ---
        t0 = time.perf_counter()
        try:
            x_cg = cal_conjugate_gradient(H, b, tol=1e-10, max_iter=10000)
            t1 = time.perf_counter()
            e = evaluate_error(x_cg, x_true, H, b)
            log(f"[CG]        time={t1-t0:.6f}s  {format_err(e)}")
            log(f"            x[:5] = {x_cg[:5]}")
            summary['CG']['time'].append(t1 - t0)
            summary['CG']['err_inf'].append(e['err_inf'])
        except Exception as ex:
            log(f"[CG]        失败: {ex}")
            summary['CG']['time'].append(None)
            summary['CG']['err_inf'].append(None)

        # --- 6. GMRES ---
        t0 = time.perf_counter()
        try:
            x_gm = cal_GMRES(H, b, tol=1e-10, max_iter=n)
            t1 = time.perf_counter()
            e = evaluate_error(x_gm, x_true, H, b)
            log(f"[GMRES]     time={t1-t0:.6f}s  {format_err(e)}")
            log(f"            x[:5] = {x_gm[:5]}")
            summary['GMRES']['time'].append(t1 - t0)
            summary['GMRES']['err_inf'].append(e['err_inf'])
        except Exception as ex:
            log(f"[GMRES]     失败: {ex}")
            summary['GMRES']['time'].append(None)
            summary['GMRES']['err_inf'].append(None)

    # ==================== 汇总 ====================
    log("")
    log("=" * 90)
    log("汇总表")
    log("=" * 90)

    ns = [10, 11, 12, 13, 14, 15]

    # 误差表
    log("")
    log("误差（无穷范数 ‖x - x*‖_∞）")
    header = f"{'n':>4} | " + " | ".join(f"{k:>12}" for k in summary)
    log(header)
    log("-" * len(header))
    for idx, n in enumerate(ns):
        row = f"{n:>4} | "
        cells = []
        for k in summary:
            v = summary[k]['err_inf'][idx]
            cells.append("   N/A" if v is None else f"{v:>12.3e}")
        row += " | ".join(cells)
        log(row)

    # 时间表
    log("")
    log("耗时（秒）")
    log(header)
    log("-" * len(header))
    for idx, n in enumerate(ns):
        row = f"{n:>4} | "
        cells = []
        for k in summary:
            v = summary[k]['time'][idx]
            cells.append("   N/A" if v is None else f"{v:>12.6f}")
        row += " | ".join(cells)
        log(row)

    out.close()
    print("\n结果已写入 linear_equation_results.txt")


if __name__ == "__main__":
    main()