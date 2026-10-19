import math


# 原方程及其导数
def f(x):
    return x**3 + 2*x**2 + 10*x - 20


def df(x):
    return 3*x**2 + 4*x + 10


# 方法(1)的迭代函数
def phi1(x):
    return (20 - 2*x**2 - x**3) / 10


# 实数立方根，避免负数开三次方变成复数
def cbrt(y):
    if y == 0:
        return 0.0
    return math.copysign(abs(y)**(1/3), y)


# 方法(2)的迭代函数
def phi2(x):
    return cbrt(20 - 10*x - 2*x**2)


# 普通不动点迭代
def fixed_point(name, phi, x0, tol=1e-8, max_iter=1000):
    x = x0
    for k in range(1, max_iter + 1):
        try:
            x_new = phi(x)
        except Exception as e:
            return name, x, k, False, f"异常: {e}"

        if not math.isfinite(x_new) or abs(x_new) > 1e12:
            return name, x, k, False, "发散或出现非有限值"

        if abs(x_new - x) < tol:
            return name, x_new, k, True, "相邻两次迭代差满足精度"

        x = x_new

    return name, x, max_iter, False, "达到最大迭代次数，未收敛"


# Steffensen 加速迭代
def steffensen(name, phi, x0, tol=1e-8, max_iter=1000):
    x = x0

    for k in range(1, max_iter + 1):
        try:
            p1 = phi(x)
            p2 = phi(p1)
            den = p2 - 2*p1 + x

            if not (math.isfinite(p1) and math.isfinite(p2) and math.isfinite(den)):
                return name, x, k, False, "出现非有限值"

            if abs(den) < 1e-14:
                return name, x, k, False, "Steffensen 分母过小"

            x_new = x - (p1 - x)**2 / den

        except Exception as e:
            return name, x, k, False, f"异常: {e}"

        if not math.isfinite(x_new) or abs(x_new) > 1e12:
            return name, x, k, False, "发散或出现非有限值"

        if abs(x_new - x) < tol:
            return name, x_new, k, True, "相邻两次迭代差满足精度"

        x = x_new

    return name, x, max_iter, False, "达到最大迭代次数，未收敛"


# Newton 法
def newton(name, x0, tol=1e-8, max_iter=100):
    x = x0

    for k in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)

        if abs(dfx) < 1e-15:
            return name, x, k, False, "导数过小"

        x_new = x - fx / dfx

        if not math.isfinite(x_new) or abs(x_new) > 1e12:
            return name, x, k, False, "发散或出现非有限值"

        if abs(x_new - x) < tol:
            return name, x_new, k, True, "相邻两次迭代差满足精度"

        x = x_new

    return name, x, max_iter, False, "达到最大迭代次数，未收敛"


def format_result(result, exact):
    """把单个方法的结果格式化为字符串列表。"""
    name, x, k, ok, msg = result
    err = abs(x - exact)
    lines = [
        name,
        f"  近似根 x = {x:.15f}",
        f"  迭代次数 = {k}",
        f"  是否收敛 = {ok}",
        f"  状态 = {msg}",
        f"  与准确解误差 = {err:.3e}",
        f"  |f(x)| = {abs(f(x)):.3e}",
        "",
    ]
    return lines


if __name__ == "__main__":
    x0 = 1.0
    exact = 1.368808107  # 题目给出的准确解
    tol = 1e-8
    output_file = "results.txt"

    results = []

    # (1) 直接迭代方法(1)
    results.append(
        fixed_point("(1) 直接迭代：x_{k+1} = (20 - 2x_k^2 - x_k^3)/10",
                    phi1, x0, tol, max_iter=1000)
    )

    # (2) 直接迭代方法(2)
    results.append(
        fixed_point("(2) 直接迭代：x_{k+1} = cbrt(20 - 10x_k - 2x_k^2)",
                    phi2, x0, tol, max_iter=1000)
    )

    # (3) 方法(1)的 Steffensen 加速
    results.append(
        steffensen("(3) 方法(1)的 Steffensen 加速",
                   phi1, x0, tol, max_iter=1000)
    )

    # (4) 方法(2)的 Steffensen 加速
    results.append(
        steffensen("(4) 方法(2)的 Steffensen 加速",
                   phi2, x0, tol, max_iter=1000)
    )

    # (5) Newton 法
    results.append(
        newton("(5) Newton 法", x0, tol, max_iter=100)
    )

    # 汇总所有输出行
    all_lines = [
        "非线性方程求根实验结果",
        "方程: x^3 + 2x^2 + 10x - 20 = 0",
        f"初值 x0 = {x0}",
        f"准确解 x* = {exact}",
        f"精度 tol = {tol}",
        "=" * 60,
        "",
    ]

    for res in results:
        all_lines.extend(format_result(res, exact))

    # 打印到屏幕
    text = "\n".join(all_lines)
    print(text)

    # 写入文件
    with open(output_file, "w", encoding="utf-8") as fout:
        fout.write(text + "\n")

    print(f"结果已写入文件：{output_file}")