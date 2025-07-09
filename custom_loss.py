import torch
import torch.nn.functional as F
from torch import nn

class SigmoidLoss(nn.Module):
    def __init__(self, adv_temperature=None):
        super().__init__()
        self.adv_temperature = adv_temperature

    def forward(self, p_scores, n_scores):
        if self.adv_temperature:
            weights = F.softmax(self.adv_temperature * n_scores, dim=-1).detach()
            n_scores = weights * n_scores
        p_loss = -F.logsigmoid(p_scores).mean()
        n_loss = -F.logsigmoid(-n_scores).mean()
        return (p_loss + n_loss) / 2, p_loss, n_loss


def nt_xent_loss(x1, x2, temperature=0.5):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)
    """
    batch_size = x1.size(0)
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    logits = torch.mm(x1, x2.t()) / temperature
    labels = torch.arange(batch_size).to(x1.device)
    loss = F.cross_entropy(logits, labels)
    return loss


def barlow_twins_loss(z1, z2, lambda_bt=0.005):
    """
    Barlow Twins Loss (Redundancy reduction)
    """
    z1 = (z1 - z1.mean(0)) / z1.std(0)
    z2 = (z2 - z2.mean(0)) / z2.std(0)
    c = torch.mm(z1.T, z2) / z1.size(0)

    on_diag = torch.diagonal(c).add_(-1).pow(2).sum()
    off_diag = off_diagonal(c).pow(2).sum()
    return on_diag + lambda_bt * off_diag


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def simclr_loss(z1, z2, temperature=0.5):
    """
    SimCLR contrastive loss
    """
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)
    similarity_matrix = torch.matmul(z, z.T)

    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(z.device)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.size(0), dtype=torch.long).to(z.device)
    logits = logits / temperature
    return F.cross_entropy(logits, labels)


def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Triplet Loss
    """
    d_ap = F.pairwise_distance(anchor, positive, p=2)
    d_an = F.pairwise_distance(anchor, negative, p=2)
    return F.relu(d_ap - d_an + margin).mean()