import torch
import torch.nn as nn

from utils.logging_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


class LossModule(nn.Module):
    def __init__(self, lambda_orthogonality, M, t):
        super(LossModule, self).__init__()

        self.lambda_orthogonality = lambda_orthogonality
        self.M = M
        self.t = t
        self.triplet_loss = nn.TripletMarginLoss(margin=M)

    def contrastive_loss(self, v, vhat, negatives):
        batch_size = vhat.size(0)
        N = negatives.size(0)
        loss = torch.zeros(batch_size, device=v.device)

        # Calculate true distance between reconstructed and real embeddings
        true_distance = self.l2(vhat, v)

        for i in range(N):  # loop over each element in "negatives"

            # Tranform negative from [embedding dim] to [batch size, embedding_dim]
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

    def focal_triplet_loss_WRONG(self, v, vhat_z, g, F):
        losses = []
        for i in range(F.size(0)):  # Iterate over each negative example
            # For each negative, compute the loss against the anchor and positive
            loss = self.triplet_loss(vhat_z, v, F[i].unsqueeze(0).expand(v.size(0), -1))
            losses.append(loss)

        loss_tensor = torch.stack(losses)
        loss = loss_tensor.mean(dim=0).mean()
        return loss

    def focal_triplet_loss(self, v, vhat_z, g, F):
        _, indices = torch.topk(g, self.t, largest=False, dim=1)

        F_t = torch.stack([F[indices[i]] for i in range(g.size(0))])

        g_tz = torch.stack([g[i, indices[i]] for i in range(g.size(0))])

        g_t = g_tz / g_tz.sum(dim=1, keepdim=True)

        # if division by zero set all nan values to 0
        g_t[torch.isnan(g_t)] = 0

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

        # Normalizing
        loss = loss / self.t
        return loss

    def orthogonality_term(self, F, reg=1e-4):
        gram_matrix = torch.mm(F, F.T)  # Compute the Gram matrix F * F^T
        identity_matrix = torch.eye(
            gram_matrix.size(0), device=gram_matrix.device
        )  # Create an identity matrix
        ortho_loss = (gram_matrix - identity_matrix).abs().sum()
        return ortho_loss

    def forward(
        self, p, a0, a1, fx, p_negatives, a0_negatives, a1_negatives, fx_negatives
    ):
        # Extract components from dictionary for predicate p
        v_p, vhat_p, d_p, g_p, F_p = p["v"], p["vhat"], p["d"], p["g"], p["F"]

        # Extract components from dictionary for ARG0
        v_a0, vhat_a0, d_a0, g_a0, F_a0 = a0["v"], a0["vhat"], a0["d"], a0["g"], a0["F"]

        # Extract components from dictionary for ARG1
        v_a1, vhat_a1, d_a1, g_a1, F_a1 = a1["v"], a1["vhat"], a1["d"], a1["g"], a1["F"]

        # Extract components from dictionary for frameaxis
        v_fx, vhat_fx, d_fx, g_fx, F_fx = fx["v"], fx["vhat"], fx["d"], fx["g"], fx["F"]

        # Calculate losses for predicate
        Ju_p = self.contrastive_loss(v_p, vhat_p, p_negatives)
        Jt_p = self.focal_triplet_loss(v_p, vhat_p, g_p, F_p)
        Jz_p = (
            Ju_p + Jt_p + self.lambda_orthogonality * self.orthogonality_term(F_p) ** 2
        )

        # Calculate losses for ARG0
        Ju_a0 = self.contrastive_loss(v_a0, vhat_a0, a0_negatives)
        Jt_a0 = self.focal_triplet_loss(v_a0, vhat_a0, g_a0, F_a0)
        Jz_a0 = (
            Ju_a0
            + Jt_a0
            + self.lambda_orthogonality * self.orthogonality_term(F_a0) ** 2
        )

        # Calculate losses for ARG1
        Ju_a1 = self.contrastive_loss(v_a1, vhat_a1, a1_negatives)
        Jt_a1 = self.focal_triplet_loss(v_a1, vhat_a1, g_a1, F_a1)
        Jz_a1 = (
            Ju_a1
            + Jt_a1
            + self.lambda_orthogonality * self.orthogonality_term(F_a1) ** 2
        )

        # Calculate losses for frameaxis
        Ju_fx = self.contrastive_loss(v_fx, vhat_fx, fx_negatives)

        # check if tensor have nan values Ju_fx
        if torch.isnan(Ju_fx).any():
            logger.debug("Ju_fx has nan")
            logger.debug("Ju_fx:", Ju_fx)

        Jt_fx = self.focal_triplet_loss(v_fx, vhat_fx, g_fx, F_fx)

        if torch.isnan(Jt_fx).any():
            logger.debug("Jt_fx has nan")
            logger.debug("Jt_fx:", Jt_fx)

        Jz_fx = (
            Ju_fx
            + Jt_fx
            + self.lambda_orthogonality * self.orthogonality_term(F_fx) ** 2
        )

        if torch.isnan(self.orthogonality_term(F_fx)).any():
            logger.debug("orthogonality_term has nan")

        if torch.isnan(Jz_p).any():
            logger.debug("Jz_p has nan")

        if torch.isnan(Jz_a0).any():
            logger.debug("Jz_a0 has nan")

        if torch.isnan(Jz_a1).any():
            logger.debug("Jz_a1 has nan")

        if torch.isnan(Jz_fx).any():
            logger.debug("Jz_fx has nan")

        # Aggregate the losses
        loss = Jz_p + Jz_a0 + Jz_a1 + Jz_fx

        return loss
