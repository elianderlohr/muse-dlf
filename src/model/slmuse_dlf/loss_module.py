from math import isnan
import torch
import torch.nn as nn
from utils.logging_manager import LoggerManager
from torch.cuda.amp import autocast


class LossModule(nn.Module):
    def __init__(self, lambda_orthogonality, M, t, _debug=False):
        super(LossModule, self).__init__()

        # init logger
        self.logger = LoggerManager.get_logger(__name__)

        self.lambda_orthogonality = lambda_orthogonality
        self.M = M
        self.t = t
        self.triplet_loss = nn.TripletMarginLoss(margin=M)

        self._debug = _debug

        # Debugging:
        self.logger.debug(f"✅ LossModule successfully initialized")

    def contrastive_loss(self, v, vhat, negatives):
        batch_size = vhat.size(0)
        N = negatives.size(0)
        loss = torch.zeros(batch_size, device=v.device)

        # Calculate true distance between reconstructed and real embeddings
        true_distance = self.l2(vhat, v)

        for i in range(N):  # loop over each element in "negatives"
            # Transform negative from [embedding dim] to [batch size, embedding_dim]
            negative = negatives[i, :].expand(v.size(0), -1)

            # Calculate negative distance for current negative embedding
            negative_distance = self.l2(vhat, negative)

            # Compute loss based on the provided logic: l2(vhat, v) + 1 + l2(vhat, negative) and clamp to 0 if below 0
            current_loss = 1 + true_distance - negative_distance
            loss += torch.clamp(current_loss, min=0.0)

        # Normalize the total loss by N
        return loss / N

    def l2(self, u, v):
        return torch.sqrt(torch.sum((u - v) ** 2, dim=1))

    def focal_triplet_loss(self, v, vhat_z, g, F):
        _, indices = torch.topk(g, self.t, largest=False, dim=1)

        F_t = torch.stack([F[indices[i]] for i in range(g.size(0))])

        g_tz = torch.stack([g[i, indices[i]] for i in range(g.size(0))])

        g_tz_sum = g_tz.sum(dim=1, keepdim=True)

        epsilon = 1e-10
        g_t = g_tz / (g_tz.sum(dim=1, keepdim=True) + epsilon)

        m_t = self.M * ((1 - g_t) ** 2)

        # Initializing loss
        loss = torch.zeros_like(v[:, 0])

        # Iteratively adding to the loss for each negative embedding
        for i in range(self.t):
            current_v_t = F_t[:, i]
            current_m_t = m_t[:, i]

            current_loss = (
                current_m_t + self.l2(vhat_z, v) - self.l2(vhat_z, current_v_t)
            )

            loss += torch.max(torch.zeros_like(current_loss), current_loss)

        # check if loss is nan or inf or 0
        if (
            torch.isnan(loss).any()
            or torch.isinf(loss).any()
            or torch.allclose(loss, torch.zeros_like(loss))
        ):
            self.logger.error("NaNs detected in focal_triplet_loss LOSS")
            self.logger.error(f"loss: {loss}")

        # Normalizing
        loss = loss / self.t

        if torch.isnan(loss).any():
            self.logger.error("NaNs detected in focal_triplet_loss")
            self.logger.error(f"v: {v}")
            self.logger.error(f"vhat_z: {vhat_z}")
            self.logger.error(f"g: {g}")
            self.logger.error(f"F: {F}")
            self.logger.error(f"loss: {loss}")

        return loss

    def orthogonality_term(self, F, reg=1e-4):
        gram_matrix = torch.mm(F, F.T)  # Compute the Gram matrix F * F^T
        identity_matrix = torch.eye(
            gram_matrix.size(0), device=gram_matrix.device
        )  # Create an identity matrix
        ortho_loss = (gram_matrix - identity_matrix).abs().sum()
        return ortho_loss

    def forward(
        self,
        c,
        negatives,
        mixed_precision="fp16",  # mixed precision as a parameter
    ):
        precision_dtype = (
            torch.float16
            if mixed_precision == "fp16"
            else torch.bfloat16 if mixed_precision == "bf16" else None
        )

        with autocast(
            enabled=mixed_precision in ["fp16", "bf16"], dtype=precision_dtype
        ):
            # Extract components from dictionary for predicate p
            v, vhat, d, g, F = c["v"], c["vhat"], c["d"], c["g"], c["F"]

            # Calculate losses for predicate
            Ju = self.contrastive_loss(v, vhat, negatives)
            Jt = self.focal_triplet_loss(v, vhat, g, F)
            Jz = Ju + Jt + self.lambda_orthogonality * self.orthogonality_term(F) ** 2

        return Jz
