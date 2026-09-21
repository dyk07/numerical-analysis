import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


# ============================================================
# 1. 基本算法实现
# ============================================================

def cg(A, b, x0=None, tol=1e-10, maxit=None, return_history=False):
    """共轭梯度法，要求 A 对称正定。"""
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    if maxit is None:
        maxit = n

    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    rs_old = r @ r

    xs = [x.copy()]
    rs = [np.linalg.norm(r)]

    for k in range(maxit):
        if np.sqrt(rs_old) < tol:
            break
        Ap = A @ p
        alpha = rs_old / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

        xs.append(x.copy())
        rs.append(np.linalg.norm(r))

    if return_history:
        return x, np.array(xs), np.array(rs)
    return x


def steepest_descent(A, b, x0=None, tol=1e-10, maxit=None, return_history=False):
    """最速下降法，用于对比。"""
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    if maxit is None:
        maxit = n * 5

    x = x0.copy()
    r = b - A @ x
    xs = [x.copy()]
    rs = [np.linalg.norm(r)]

    for k in range(maxit):
        if np.linalg.norm(r) < tol:
            break
        Ar = A @ r
        alpha = (r @ r) / (r @ Ar)
        x = x + alpha * r
        r = r - alpha * Ar
        xs.append(x.copy())
        rs.append(np.linalg.norm(r))

    if return_history:
        return x, np.array(xs), np.array(rs)
    return x


def gmres(A, b, x0=None, tol=1e-10, maxit=None, restart=None,
          return_history=False):
    """
    GMRES(m) 方法。restart=None 表示不重启。
    返回残差历史（由 Givens 旋转得到的真实残差范数）。
    """
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    if maxit is None:
        maxit = n
    if restart is None:
        restart = maxit

    x = x0.copy()
    rs = [np.linalg.norm(b - A @ x)]
    total_iter = 0

    while total_iter < maxit:
        r = b - A @ x
        beta = np.linalg.norm(r)
        if beta < tol:
            break

        m = min(restart, maxit - total_iter)
        V = np.zeros((n, m + 1))
        H = np.zeros((m + 1, m))
        V[:, 0] = r / beta

        cs = np.zeros(m)
        sn = np.zeros(m)
        g = np.zeros(m + 1)
        g[0] = beta

        for j in range(m):
            w = A @ V[:, j]
            for i in range(j + 1):
                H[i, j] = V[:, i] @ w
                w = w - H[i, j] * V[:, i]
            H[j + 1, j] = np.linalg.norm(w)
            if H[j + 1, j] > 1e-14:
                V[:, j + 1] = w / H[j + 1, j]

            # 应用之前的 Givens 旋转到第 j 列
            for i in range(j):
                temp = cs[i] * H[i, j] + sn[i] * H[i + 1, j]
                H[i + 1, j] = -sn[i] * H[i, j] + cs[i] * H[i + 1, j]
                H[i, j] = temp

            # 计算新的 Givens 旋转，消去 H[j+1, j]
            denom = np.hypot(H[j, j], H[j + 1, j])
            cs[j] = H[j, j] / denom
            sn[j] = H[j + 1, j] / denom
            H[j, j] = cs[j] * H[j, j] + sn[j] * H[j + 1, j]
            H[j + 1, j] = 0.0

            # 更新右端项 g
            temp = cs[j] * g[j] + sn[j] * g[j + 1]
            g[j + 1] = -sn[j] * g[j] + cs[j] * g[j + 1]
            g[j] = temp

            total_iter += 1
            rs.append(abs(g[j + 1]))
            if abs(g[j + 1]) < tol:
                break

        # 回代求解 R y = g(1:j+1)
        k = min(j + 1, m)
        y = np.zeros(k)
        R = H[:k, :k]
        for i in range(k - 1, -1, -1):
            y[i] = (g[i] - R[i, i + 1:k] @ y[i + 1:k]) / R[i, i]
        x = x + V[:, :k] @ y

        if rs[-1] < tol:
            break

    if return_history:
        return x, np.array(rs)
    return x


# ============================================================
# 2. 实验设置
# ============================================================

np.random.seed(0)

# 构造一个二维对称正定矩阵，便于可视化
A2 = np.array([[4.0, 1.0],
               [1.0, 3.0]])
b2 = np.array([1.0, 2.0])
x_exact2 = np.linalg.solve(A2, b2)

# 构造一个较大规模的非对称矩阵，用于 GMRES
n = 50
D = np.diag(np.linspace(1, 100, n))
Q, _ = np.linalg.qr(np.random.randn(n, n))
A_big = Q @ D @ Q.T          # 对称正定
A_nonsym = A_big + 5.0 * np.triu(np.random.randn(n, n), 1)
b_big = np.random.randn(n)


# ============================================================
# 3. 图 1：CG 在二维二次泛函上的几何轨迹
# ============================================================

x_cg, xs_cg, rs_cg = cg(A2, b2, x0=np.array([-2.0, 2.5]),
                        return_history=True)
x_sd, xs_sd, rs_sd = steepest_descent(A2, b2, x0=np.array([-2.0, 2.5]),
                                      return_history=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# 左图：等高线 + 迭代点
ax = axes[0]
gx = np.linspace(-3.0, 3.0, 400)
gy = np.linspace(-3.0, 3.0, 400)
GX, GY = np.meshgrid(gx, gy)
Z = 0.5 * (A2[0, 0] * GX**2 + 2 * A2[0, 1] * GX * GY + A2[1, 1] * GY**2) \
    - b2[0] * GX - b2[1] * GY

cs = ax.contour(GX, GY, Z, levels=25, cmap='viridis', linewidths=0.9)
ax.clabel(cs, inline=True, fontsize=7)
ax.plot(x_exact2[0], x_exact2[1], 'k*', markersize=16,
        label='精确解 $x^*$')
ax.plot(xs_cg[:, 0], xs_cg[:, 1], 'o-', color='crimson',
        linewidth=1.8, markersize=6, label='CG 迭代轨迹')
ax.plot(xs_sd[:8, 0], xs_sd[:8, 1], 's--', color='navy',
        linewidth=1.2, markersize=5, label='最速下降（前 8 步）')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('CG 与最速下降法在二次泛函等高线上的轨迹')
ax.legend(loc='upper right', fontsize=9)
ax.set_aspect('equal')

# 右图：残差收敛曲线
ax = axes[1]
ax.semilogy(rs_cg, 'o-', color='crimson', label='CG')
ax.semilogy(rs_sd[:15], 's--', color='navy', label='最速下降')
ax.set_xlabel('迭代步 $k$')
ax.set_ylabel(r'$\|r_k\|_2$')
ax.set_title('二维问题上 CG 与最速下降的残差')
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('cg_2d_demo.png', dpi=150)
plt.show()


# ============================================================
# 4. 图 2：大规模问题上 CG 与 GMRES 的收敛对比
# ============================================================

# CG 在对称正定矩阵上
x_cg_big, rs_cg_big = cg(A_big, b_big, tol=1e-10, maxit=n,
                          return_history=True)

# GMRES 不重启
x_gm_big, rs_gm_big = gmres(A_nonsym, b_big, tol=1e-10,
                            maxit=80, restart=None,
                            return_history=True)

# GMRES(20) 重启
x_gm_r, rs_gm_r = gmres(A_nonsym, b_big, tol=1e-10,
                        maxit=80, restart=20,
                        return_history=True)

x_direct = np.linalg.solve(A_big, b_big)
err_cg = np.linalg.norm(x_cg_big - x_direct) / np.linalg.norm(x_direct)
err_gm = np.linalg.norm(x_gm_big - x_direct) / np.linalg.norm(x_direct)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
ax.semilogy(rs_cg_big, 'o-', color='crimson', label='CG（SPD 矩阵）')
ax.semilogy(rs_gm_big, 's-', color='seagreen',
            label='GMRES（非对称矩阵，不重启）')
ax.semilogy(rs_gm_r, '^--', color='darkorange',
            label='GMRES(20)（非对称矩阵，重启）')
ax.axhline(1e-10, color='gray', linestyle=':', label='容差 $10^{-10}$')
ax.set_xlabel('迭代步 $k$')
ax.set_ylabel(r'$\|r_k\|_2$')
ax.set_title('CG 与 GMRES 的残差收敛曲线')
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=9)

# 右图：条件数与特征值分布
eig_A = np.linalg.eigvalsh(A_big)
eig_ns = np.linalg.eigvals(A_nonsym)
ax = axes[1]
ax.plot(eig_A.real, eig_A.imag, 'o', color='crimson',
        label=f'$A$（SPD），$\\kappa={eig_A[-1]/eig_A[0]:.1f}$')
ax.plot(eig_ns.real, eig_ns.imag, 's', color='seagreen',
        label='非对称矩阵 $A_{ns}$ 的特征值')
ax.axhline(0, color='k', linewidth=0.6)
ax.axvline(0, color='k', linewidth=0.6)
ax.set_xlabel(r'Re$(\lambda)$')
ax.set_ylabel(r'Im$(\lambda)$')
ax.set_title('特征值分布')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('cg_gmres_convergence.png', dpi=150)
plt.show()


# ============================================================
# 5. 数值验证输出
# ============================================================

print('=' * 60)
print('二维问题')
print('  精确解      :', x_exact2)
print('  CG 解       :', x_cg)
print('  CG 步数     :', len(rs_cg) - 1)
print('  最速下降步数:', len(rs_sd) - 1)
print()
print('50 维问题')
print('  CG 相对误差  :', err_cg)
print('  GMRES 相对误差:', err_gm)
print('  CG 迭代步数  :', len(rs_cg_big) - 1)
print('  GMRES 迭代步数:', len(rs_gm_big) - 1)
print('=' * 60)