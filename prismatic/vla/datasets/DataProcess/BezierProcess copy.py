import numpy as np

# import sys, os
# # 将项目根目录加入模块搜索路径
# root = os.path.abspath(os.path.join(__file__, "../../../../.."))
# sys.path.append(root)
import torch
from prismatic.vla.constants import TOKEN_SEQUENCE_LINE


class fitBezierToolBox:
    
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
    
    # 最小二乘求解中间控制点
    @staticmethod 
    def fit_quadratic_bezier(points):
        # Solve for best-fit P1 between fixed P0 and P2
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().to(torch.float32).numpy()

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
        if ((len(points) <= TOKEN_SEQUENCE_LINE) and err <= epsilon) or len(points) < 3: # Define Max Curve 
            return [bezier.BezierPram_to_array()]
        # split at worst point
        left_pts  = points[:idx+1]
        right_pts = points[idx:]
        left_curves  = fitBezierToolBox.fit_beziers(left_pts, epsilon)
        right_curves = fitBezierToolBox.fit_beziers(right_pts, epsilon)
        return left_curves + right_curves
    
    @staticmethod
    def compute_curve_loss(curve, points):
        """
        计算一段 Bézier 曲线在给定 points 上的累计欧氏距离误差
        curve: array-like, shape (3, D)  三个控制点 P0, P1, P2
        points: array-like, shape (N, D)  原始采样点
        返回: scalar, sum of L2 distances
        """
        
        error = 0.0
        length = len(points)
        for idx, point in enumerate(points):
            t = (idx + 1) / length
            u = 1 - t
            bezier_point = (u*u)*curve[0] + 2*u*t*curve[1] + (t*t)*curve[2]
            # 直接对 1D 向量求范数
            error += np.linalg.norm(bezier_point - point)
        return error

    @staticmethod
    def compute_loss(batch_curves, batch_points):
        """
        batch_curves: list of lists, 每个 batch 样本包含多段 curve，
                      curve 格式为 [P0, P1, P2, segment_length]
        batch_points: list of arrays, 每个 batch 样本的原始采样点 array(N, D)
        返回: 全批次的平均误差
        """
        batch_errors = []
        for curves, points in zip(batch_curves, batch_points):
            if isinstance(points, list):
                points = np.array(points, dtype=float)
            sample_error = 0.0
            consumed_pts = 0
            # 遍历每一段 Bézier curve
            for curve in curves:
                seg_len = int(curve[3])  # 第 2 项是本段应该包含的 sample 数
                start = consumed_pts
                end = consumed_pts + seg_len

                # 先切片
                seg_points = points[start:end]  # 真实可能不足
                real_len = seg_points.shape[0]

                # 如果不够长，就补零
                if real_len < seg_len:
                    # points 的第二维是动作维度 D
                    D = points.shape[1]
                    pad = np.zeros((seg_len - real_len, D), dtype=points.dtype)
                    seg_points = np.concatenate([seg_points, pad], axis=0)
                consumed_pts += seg_len
                sample_error += fitBezierToolBox.compute_curve_loss(curve[:3], seg_points) #return list(self.linePoints()) + [self.length]
            # 归一化：对整个样本除以总点数
            sample_error /= consumed_pts
            batch_errors.append(sample_error)

        # 返回全批次的平均
        return float(np.mean(batch_errors))    
    

class QuadraticBezier(fitBezierToolBox):
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

    @staticmethod
    def PointsToBezierPram(points):
        # 从途径坐标点初始化贝塞尔曲线
        bezier,_ = fitBezierToolBox.fit_quadratic_bezier(points)
        return bezier
    
    @staticmethod
    def PointsToBezierPram(P0,P1,P2):
        bezier,_ = fitBezierToolBox.fit_quadratic_bezier([P0,P1,P2])
        return bezier
    
    def linePoints(self):
        return self.P0,self.point(self, 0.5),self.P2

    def point(self, t):
        """
        计算曲线在参数 t 处的空间坐标。
        B(t) = (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2
        """
        u = 1 - t
        return u*u*self.P0 + 2*u*t*self.P1 + t*t*self.P2

    def BezierPram_to_array(self):
        """
        返回控制点和时间长度的数组，形状为 (4, D)：
        - 前三行：控制点 P0, P1, P2 的坐标，每行长度为 D。
        - 第四行：时间长度，使用 np.full 创建，与控制点坐标维度一致，
          表示该曲线段包含的动作（点）数。
        """
        # 将三个控制点堆叠为 (3, D) 形式
        coords = [self.P0, self.P1, self.P2, self.length] # (3, D)
        return coords
    
    def Bezier_to_Points(self):
        # points: list of (D,) 向量
        pts = np.array(self.linePoints())      # shape (N, D)
        flat_pts = pts.reshape(-1)             # shape (N*D,)
        return np.concatenate([flat_pts, [self.length]])

    



