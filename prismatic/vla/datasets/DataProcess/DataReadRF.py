import numpy as np

# 定义三点贝塞尔曲线在t处的值
def evaluate_quadratic(bezier, t):
    """计算三点贝塞尔曲线在参数t处的点
    Args:
        bezier (list): [P0, P1, P2], 每个是np.array(shape=(d,))
        t (float): 参数，0 <= t <= 1
    Returns:
        np.array(shape=(d,)): B(t)
    """
    P0, P1, P2 = [np.array(p) for p in bezier]
    return (1 - t)**2 * P0 + 2*(1 - t)*t * P1 + t**2 * P2

# 拟合单个三点贝塞尔曲线到给定点集
def fit_quadratic_bezier(points):
    """使用最小二乘法拟合三点贝塞尔曲线到给定点集
    Args:
        points (np.array): shape=(m,d), m个点，每点d维
    Returns:
        list: [P0,P1,P2], 每个是np.array(shape=(d,))
    """
    points = np.array(points)
    if points.ndim == 1:
        points = points[:, np.newaxis]
    m, d = points.shape
    if m < 2:
        return None
    P0 = points[0]
    P2 = points[-1]
    if m == 2:
        P1 = (P0 + P2) / 2
    else:
        t = np.linspace(0, 1, m)
        a = (1 - t)**2
        b = 2 * (1 - t) * t
        c = t**2
        d = points - a[:, np.newaxis] * P0 - c[:, np.newaxis] * P2
        sum_b2 = np.sum(b**2)
        sum_bd = np.sum(b[:, np.newaxis] * d, axis=0)
        if sum_b2 > 0:
            P1 = sum_bd / sum_b2
        else:
            P1 = (P0 + P2) / 2
    return [P0, P1, P2]

# 计算最大误差
def compute_max_error(points, first, last, bezier):
    """计算给定贝塞尔曲线与点集之间的最大欧氏距离误差
    Args:
        points (np.array): shape=(n,d)
        first (int): 起始索引
        last (int): 结束索引
        bezier (list): [P0,P1,P2]
    Returns:
        float: 最大误差
    """
    max_err = 0
    for i in range(first + 1, last):
        t = (i - first) / (last - first)
        B_t = evaluate_quadratic(bezier, t)
        err = np.linalg.norm(points[i] - B_t)
        if err > max_err:
            max_err = err
    return max_err

# 找到最大误差点
def find_split_point(points, first, last, bezier):
    """找到误差最大的点
    Args:
        points (np.array): shape=(n,d)
        first (int): 起始索引
        last (int): 结束索引
        bezier (list): [P0,P1,P2]
    Returns:
        int: 分割点索引
    """
    max_err = 0
    split = first + 1
    for i in range(first + 1, last):
        t = (i - first) / (last - first)
        B_t = evaluate_quadratic(bezier, t)
        err = np.linalg.norm(points[i] - B_t)
        if err > max_err:
            max_err = err
            split = i
    return split

# 递归拟合多个贝塞尔曲线
def fit_segments(points, first, last, error, segments_list):
    """递归地使用贪心算法拟合多个三点贝塞尔曲线
    Args:
        points (np.array): shape=(n,d)
        first (int): 当前段的起始索引
        last (int): 当前段的结束索引
        error (float): 允许的最大误差
        segments_list (list): 存储段的列表
    """
    if first >= last:
        return
    subset = points[first:last+1]
    bezier = fit_quadratic_bezier(subset)
    if bezier is None:
        return
    max_err = compute_max_error(points, first, last, bezier)
    if max_err < error:
        segments_list.append((first, last))
    else:
        split = find_split_point(points, first, last, bezier)
        fit_segments(points, first, split, error, segments_list)
        fit_segments(points, split, last, error, segments_list)

# 主函数1: 使用单个三点贝塞尔曲线拟合整个序列，每个维度单独拟合
def fit_single_quadratic(tokens):
    """使用单个三点贝塞尔曲线拟合整个动作token序列，每个维度单独拟合
    Args:
        tokens (list or np.array): shape=(n,d), n个token，每个d维
    Returns:
        list: [P0,P1,P2], 每个是np.array(shape=(d,))
    """
    tokens = np.array(tokens)
    n, d = tokens.shape
    if n < 2:
        return None
    control_points = []
    for dim in range(d):
        points_dim = tokens[:, dim]
        cp_dim = fit_quadratic_bezier(points_dim)
        control_points.append(cp_dim)
    # 转置
    bezier_3d = [[cp[0] for cp in control_points], [cp[1] for cp in control_points], [cp[2] for cp in control_points]]
    bezier_3d = [np.array(bc) for bc in bezier_3d]
    return bezier_3d

# 主函数2: 使用贪心算法拟合多个三点贝塞尔曲线
def fit_multiple_quadratic(tokens, error=0.1):
    """使用贪心算法将动作token序列拟合为多个三点贝塞尔曲线
    Args:
        tokens (list or np.array): shape=(n,d)
        error (float): 允许的最大误差
    Returns:
        list: 每个元素是([P0,P1,P2], first, last), [P0,P1,P2]是np.array(shape=(d,))
    """
    tokens = np.array(tokens)
    n, d = tokens.shape
    if n < 2:
        return []
    segments_list = []
    fit_segments(tokens, 0, n-1, error, segments_list)
    all_beziers = []
    for first, last in segments_list:
        subset = tokens[first:last+1]
        bezier = fit_quadratic_bezier(subset)
        all_beziers.append((bezier, first, last))
    return all_beziers

# 计算总平方误差
def compute_total_squared_error(all_beziers, tokens):
    """计算拟合的贝塞尔曲线与原token序列之间的总平方误差
    Args:
        all_beziers (list): 从fit_multiple_quadratic返回的列表
        tokens (np.array): shape=(n,d)
    Returns:
        float: 总平方误差
    """
    total_error = 0
    for bezier, first, last in all_beziers:
        m = last - first + 1
        for i in range(m):
            t = i / (m - 1)
            B_t = evaluate_quadratic(bezier, t)
            err = tokens[first + i] - B_t
            total_error += np.sum(err**2)
    return total_error

# # loss函数设计
# def compute_loss(all_beziers, tokens, lambda_val=1.0):
#     """计算损失函数，兼顾误差和贝塞尔曲线数量
#     Args:
#         all_beziers (list): 从fit_multiple_quadratic返回的列表
#         tokens (np.array): shape=(n,d)
#         lambda_val (float): 权重系数
#     Returns:
#         float: loss值
#     """
#     total_error = compute_total_squared_error(all_beziers, tokens)
#     num_beziers = len(all_beziers)
#     loss = total_error + lambda_val * num_beziers
#     return loss
