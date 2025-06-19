import numpy as np


class QuadraticBezier:
    """
    二次贝塞尔曲线类，包含三个控制点和时间长度（动作数）。
    """
    def __init__(self, P0, P1, P2, length):
        # 控制点转换为 NumPy 数组
        self.P0 = np.array(P0, dtype=float)
        self.P1 = np.array(P1, dtype=float)
        self.P2 = np.array(P2, dtype=float)
        # 时间长度：对应原始序列中动作点数
        self.length = length

    def point(self, t):
        """
        计算曲线在参数 t 处的空间坐标。
        B(t) = (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2
        """
        u = 1 - t
        return u*u*self.P0 + 2*u*t*self.P1 + t*t*self.P2
    
    # def split(self, t):
    #     Q0 = (1-t)*self.P0 + t*self.P1
    #     Q1 = (1-t)*self.P1 + t*self.P2
    #     R  = (1-t)*Q0      + t*Q1
    #     left  = QuadraticBezier(self.P0, Q0, R, length=None)
    #     right = QuadraticBezier(R, Q1, self.P2, length=None)
    #     return left, right

    def to_array(self):
        """
        返回控制点和时间长度的数组，形状为 (4, D)：
        - 前三行：控制点 P0, P1, P2 的坐标，每行长度为 D。
        - 第四行：时间长度，使用 np.full 创建，与控制点坐标维度一致，
          表示该曲线段包含的动作（点）数。
        """
        # 将三个控制点堆叠为 (3, D) 形式
        coords = [self.P0, self.P1, self.P2, self.length] # (3, D)
        # D = coords.shape[1]
        # # 创建一行形状 (1, D) 的数组，所有元素都等于 self.length
        # # 代表该段在时间轴上跨越的动作数量
        # length_row = np.full((1, D), self.length)
        # # 将控制点数组与长度行垂直合并，得到 (4, D)
        # return np.vstack([coords, length_row])
        return coords

class fitBezierToolBox:

    # # 定义三点贝塞尔曲线在t处的值
    # def evaluate_quadratic(self,bezier, t):
    #     """计算三点贝塞尔曲线在参数t处的点
    #     Args:
    #         bezier (list): [P0, P1, P2], 每个是np.array(shape=(d,))
    #         t (float): 参数，0 <= t <= 1
    #     Returns:
    #         np.array(shape=(d,)): B(t)
    #     """
    #     P0, P1, P2 = [np.array(p) for p in bezier]
    #     return (1 - t)**2 * P0 + 2*(1 - t)*t * P1 + t**2 * P2
    
    # 动作转换为曲线
    @staticmethod
    def dots_into_bezier(dots):
        length = len(dots)
        if length < 3:
            return print("dots is less then 3, fun: dots_into_bezier ")
        index = 0
        beziers = []
        while index  < length:
            if index + 2 == length:
                P0,P1 = dots[index:index+2] 
                P2 = P1
            elif index + 1 == length:
                P0 = dots[index] 
                P1 = P0
                P2 = P1
            else:
                P0,P1,P2 = dots[index:index+3]
            P1 = - (0.25 * P0 + 0.25 * P2 - P1) * 2
            line = [P0,P1,P2]
            beziers.append(line)
            index += 1

        return beziers
    
    # 参数化
    # @staticmethod
    # def chord_length_parameterize(points):
    #     pts = np.array(points, dtype=float)
    #     N = len(pts)
    #     dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    #     t = np.zeros(N)
    #     t[1:] = np.cumsum(dists)
    #     t /= t[-1]
    #     return t
    
    # 最小二乘求解中间控制点
    @staticmethod 
    def fit_quadratic_bezier(points):
        # Solve for best-fit P1 between fixed P0 and P2
        pts = np.array(points, dtype=float)
        N = len(pts)
        P0, P2 = pts[0], pts[-1]


        t = [i for i in range(N)]       # build weights
        t = np.array(t, dtype=float)
        t /= (N - 1)
        alpha = (1 - t)**2
        beta  = 2 * (1 - t) * t
        gamma = t**2

        if N == 2:# only 2 dots creat a line
            P1 = (P0 + P2)*0.5
            return QuadraticBezier(P0, P1, P2, N  - 1), t

        # least squares for P1: sum beta_i^2 P1 = sum beta_i (Pi - alpha_i P0 - gamma_i P2)
        B = beta
        C = alpha[:, None]*P0 + gamma[:, None]*P2
        numerator   = np.sum((B[:, None] * (pts - C)), axis=0)
        raw_den = np.sum(beta**2)
        denominator = raw_den if raw_den != 0 else 1e-9
        P1 = numerator / denominator # 梯度为0的点
        return QuadraticBezier(P0, P1, P2, N  - 1), t
    
    @staticmethod 
    def max_error(bezier, points, t):
        pts = np.array(points, dtype=float)
        fitted = np.array([bezier.point(ti) for ti in t])
        errors = np.linalg.norm(fitted - pts, axis=1)#欧几里得距离
        idx = np.argmax(errors)
        return errors[idx], idx
    
    @staticmethod
    def fit_beziers(points, epsilon):
        # print("fit_beziers: " + str(len(points)))
        """
        Recursively fits points with quadratic Beziers until max error <= epsilon.
        Returns list of QuadraticBezier instances.
        """
        bezier, t = fitBezierToolBox.fit_quadratic_bezier(points)
        err, idx = fitBezierToolBox.max_error(bezier, points, t)
        # print("err: "+ str(err))
        # print("idx: "+ str(idx))
        if err <= epsilon or len(points) < 3:
            return [bezier.to_array()]
        # split at worst point
        left_pts  = points[:idx+1]
        right_pts = points[idx:]
        left_curves  = fitBezierToolBox.fit_beziers(left_pts, epsilon)
        right_curves = fitBezierToolBox.fit_beziers(right_pts, epsilon)
        return left_curves + right_curves
    
    