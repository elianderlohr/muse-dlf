from math import isnan
import torch
import torch.nn as nn
from utils.logging_manager import LoggerManager
from torch.cuda.amp import autocast


class LossModule(nn.Module):
    def __init__(self, lambda_orthogonality, M, t, _debug=False):
        super(LossModule, self).__init__()
        self.logger = LoggerManager.get_logger(__name__)
        self.lambda_orthogonality = lambda_orthogonality
        self.M = M
        self.t = t
        self.triplet_loss = nn.TripletMarginLoss(margin=M)
        self._debug = _debug
        self.logger.debug(f"✅ LossModule successfully initialized")

    def contrastive_loss(self, v, vhat, negatives, mask):
        batch_size = vhat.size(0)
        N = negatives.size(0)
        loss = torch.zeros(batch_size, device=v.device)
        true_distance = self.l2(vhat, v)

        for i in range(N):
            negative = negatives[i, :].expand(v.size(0), -1)
            negative_distance = self.l2(vhat, negative)
            current_loss = 1 + true_distance - negative_distance
            loss += torch.clamp(current_loss, min=0.0)

        loss = loss / N
        loss = loss * mask.float()
        if mask.sum() == 0:
            return torch.tensor(0.0, device=loss.device)
        return loss.sum() / mask.sum()

    def focal_triplet_loss(self, v, vhat_z, g, F, mask):
        _, indices = torch.topk(g, self.t, largest=False, dim=1)
        F_t = torch.stack([F[indices[i]] for i in range(g.size(0))])
        g_tz = torch.stack([g[i, indices[i]] for i in range(g.size(0))])
        g_tz_sum = g_tz.sum(dim=1, keepdim=True)
        epsilon = 1e-10
        g_t = g_tz / (g_tz.sum(dim=1, keepdim=True) + epsilon)
        m_t = self.M * ((1 - g_t) ** 2)
        loss = torch.zeros_like(v[:, 0])

        for i in range(self.t):
            current_v_t = F_t[:, i]
            current_m_t = m_t[:, i]
            current_loss = (
                current_m_t + self.l2(vhat_z, v) - self.l2(vhat_z, current_v_t)
            )
            loss += torch.max(torch.zeros_like(current_loss), current_loss)

        loss = loss * mask.float()
        if mask.sum() == 0:
            return torch.tensor(0.0, device=loss.device)
        return loss.sum() / mask.sum()

    def forward(
        self,
        c,
        negatives,
        mask,
        mixed_precision="fp16",
    ):
        precision_dtype = (
            torch.float16
            if mixed_precision == "fp16"
            else torch.bfloat16 if mixed_precision == "bf16" else torch.float32
        )

        with autocast(
            enabled=mixed_precision in ["fp16", "bf16", "fp32"], dtype=precision_dtype
        ):
            v, vhat, d, g, F = c["v"], c["vhat"], c["d"], c["g"], c["F"]
            Ju = self.contrastive_loss(v, vhat, negatives, mask)
            Jt = self.focal_triplet_loss(v, vhat, g, F, mask)
            Jz = Ju + Jt + self.lambda_orthogonality * self.orthogonality_term(F) ** 2

        return Jz
