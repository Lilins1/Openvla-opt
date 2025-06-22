import torch
from prismatic.vla.constants import TOKEN_SEQUENCE_LINE
from prismatic.vla.constants import DEBUG

class fitBezierToolBox:
    
    @staticmethod
    def Debug(a:str,b):
        if DEBUG == True:
            print(a + ": " + str(b))

    @staticmethod
    def dots_into_bezier(dots):
        length = len(dots)
        if length < 3:
            print("dots is less than 3, fun: dots_into_bezier")
            return []
        beziers = []
        index = 0
        while index < length:
            if index + 2 == length:
                P0, P1 = dots[index:index+2]
                P2 = P1.clone()
            elif index + 1 == length:
                P0 = dots[index]
                P1 = P0.clone()
                P2 = P1.clone()
            else:
                P0, P1, P2 = dots[index:index+3]
                P1 = - (0.25 * P0 + 0.25 * P2 - P1) * 2
            beziers.append([P0, P1, P2])
            index += 1
        return beziers

    @staticmethod
    def fit_quadratic_bezier(points):
        if isinstance(points, list):
            points = torch.stack(points)
        device = points.device
        N = points.shape[0]
        P0 = points[0]
        P2 = points[-1]
        t = torch.linspace(0, 1, N, device=device)
        alpha = (1 - t)**2
        beta = 2 * (1 - t) * t
        gamma = t**2
        if N == 2:
            P1 = (P0 + P2) * 0.5
        else:
            C = alpha[:, None] * P0 + gamma[:, None] * P2
            numerator = torch.sum(beta[:, None] * (points - C), dim=0)
            denominator = torch.sum(beta**2)
            P1 = numerator / denominator if denominator != 0 else torch.zeros_like(P0)
        return [P0, P1, P2, N - 1], t

    @staticmethod
    def max_error(bezier, points, t):
        bezier_points = fitBezierToolBox.bezier_point(bezier, t)
        errors = torch.norm(bezier_points - points, dim=1)
        idx = torch.argmax(errors)
        return errors[idx], idx.item()

    @staticmethod
    def bezier_point(bezier, t):
        P0, P1, P2, _ = bezier
        u = 1 - t
        return u[:, None]**2 * P0 + 2 * u[:, None] * t[:, None] * P1 + t[:, None]**2 * P2

    @staticmethod
    def fit_beziers(points, epsilon):
        if len(points) < 3:
            return []
        bezier, t = fitBezierToolBox.fit_quadratic_bezier(points)
        err, idx = fitBezierToolBox.max_error(bezier, points, t)
        if (len(points) <= TOKEN_SEQUENCE_LINE and err <= epsilon) or len(points) < 3:
            return [bezier]
         # ⚠️ 添加防止无限递归的条件
        if idx <= 0 or idx >= len(points) - 1:
            idx = len(points)//2
        left_pts = points[:idx+1]
        right_pts = points[idx:]
        left_curves = fitBezierToolBox.fit_beziers(left_pts, epsilon)
        right_curves = fitBezierToolBox.fit_beziers(right_pts, epsilon)
        return left_curves + right_curves

    @staticmethod
    def compute_curve_loss(combined_curve, points, pt_dim):
        """
        combined_curve: Tensor of shape (pt_dim*3 + 1,), last element is segment length
        points: Tensor of shape (seg_len, D)
        """
        # unpack
        P0 = combined_curve[:pt_dim]
        P1 = combined_curve[pt_dim:2*pt_dim]
        P2 = combined_curve[2*pt_dim:3*pt_dim]
        # length = combined_curve[-1]
        N = points.shape[0]
        t = torch.linspace(0, 1, N, device=points.device)
        # generate bezier points: implement bezier formula directly or via toolbox
        # Assuming fitBezierToolBox.bezier_point accepts control tuple and t
        bezier_points = fitBezierToolBox.bezier_point((P0, P1, P2,combined_curve[-1]), t)
        # fitBezierToolBox.Debug("bezier_points",bezier_points)
        # fitBezierToolBox.Debug("points",points)

        error = torch.sum(torch.norm(bezier_points - points, dim=1))
        return error

    @staticmethod
    def compute_loss(combined_curves, batch_points, pt_dim):
        """
        combined_curves: Tensor of shape (B, seq_len, pt_dim*3+1)
        batch_points: Tensor of shape (B, max_points, D)
        pt_dim: dimension of each control point
        """
        batch_size, seq_len, _ = combined_curves.shape
        batch_errors = []
        for i in range(batch_size):
            sample_curves = combined_curves[i]  # (seq_len, 3*pt_dim+1)
            points = batch_points[i]
            sample_error = 0.0
            consumed = 0
            for curve in sample_curves:
                seg_len = int(curve[-1].item())
                seg_pts = points[consumed:consumed+seg_len]
                real_len = seg_pts.shape[0]
                if real_len < seg_len:
                    pad = torch.zeros((seg_len-real_len, points.shape[1]), device=points.device)
                    seg_pts = torch.cat([seg_pts, pad], dim=0)
                sample_error += fitBezierToolBox.compute_curve_loss(curve, seg_pts, pt_dim)
                consumed += seg_len
            sample_error = sample_error / consumed if consumed > 0 else 0.0
            batch_errors.append(sample_error)
        return torch.mean(torch.stack(batch_errors))
    
    @staticmethod
    def make_ground_truth_tensors(ground_truth_curves, device, pt_dim, seq_len):
        """
        Convert list-of-curves into a tensor of shape (batch_size, seq_len, 4, pt_dim)
        ground_truth_curves: List[batch] where each element is a list of seq_len segments,
                              each segment is [P0, P1, P2, length]
        """
        batch_size = len(ground_truth_curves)
        tensor_curves = torch.zeros((batch_size, seq_len, 4, pt_dim), device=device)
        for i, curves in enumerate(ground_truth_curves):
            # curves is a list of at most seq_len segments
            for j, (P0, P1, P2, length) in enumerate(curves[:seq_len]):
                tensor_curves[i, j, 0, :] = P0.to(device).float()
                tensor_curves[i, j, 1, :] = P1.to(device).float()
                tensor_curves[i, j, 2, :] = P2.to(device).float()
                tensor_curves[i, j, 3, 0] = float(length)
        return tensor_curves
    

    @staticmethod
    def curves_to_combined(batch_curves):
        """
        Convert batch_curves of shape (B, seq_len, 4, pt_dim)
        into combined format (B, seq_len, pt_dim*3 + 1)
        where last dim stores P0, P1, P2, and length scalar.
        """
        P0 = batch_curves[..., 0, :]
        P1 = batch_curves[..., 1, :]
        P2 = batch_curves[..., 2, :]
        lengths = batch_curves[..., 3, 0].unsqueeze(-1)
        combined = torch.cat([P0, P1, P2, lengths], dim=-1)
        return combined

class QuadraticBezier(fitBezierToolBox):
    def __init__(self, P0, P1, P2, length, device='cuda'):
        self.P0 = torch.tensor(P0, dtype=torch.float32, device=device)
        self.P1 = torch.tensor(P1, dtype=torch.float32, device=device)
        self.P2 = torch.tensor(P2, dtype=torch.float32, device=device)
        self.length = length

    @staticmethod
    def PointsToBezierPram(points):
        bezier, _ = fitBezierToolBox.fit_quadratic_bezier(points)
        P0, P1, P2, length = bezier
        return QuadraticBezier(P0, P1, P2, length, device=points.device)

    @staticmethod
    def PointsToBezierPram(P0, P1, P2):
        points = torch.stack([P0, P1, P2])
        bezier, _ = fitBezierToolBox.fit_quadratic_bezier(points)
        P0, P1, P2, length = bezier
        return QuadraticBezier(P0, P1, P2, length, device=P0.device)

    def point(self, t):
        u = 1 - t
        return u*u*self.P0 + 2*u*t*self.P1 + t*t*self.P2

    def BezierPram_to_array(self):
        return [self.P0, self.P1, self.P2, self.length]

    def Bezier_to_Points(self):
        t = torch.tensor([0.0, 0.5, 1.0], device=self.P0.device)
        points = self.point(t)
        return torch.cat([points.flatten(), torch.tensor([self.length], device=self.P0.device)])