"""CSD/SRD and task loss; explicit, auditable interpretation of Eqs. (35)-(42)."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from aumdf.model import MODALITIES


def freeze_teacher(model):
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)


class DistillationLoss(nn.Module):
    def __init__(self, hidden_dim, lambda_reg=0.1, lambda_csd=0.1, lambda_srd=0.1,
                 qkv_coefficient=1e-4, temperature=0.07):
        super().__init__()
        self.lambda_reg, self.lambda_csd, self.lambda_srd = lambda_reg, lambda_csd, lambda_srd
        self.qkv_coefficient = qkv_coefficient
        self.raw_scales = nn.Parameter(torch.full((2, 3, 2), math.log(math.expm1(1 / temperature))))
        self.teacher_projection = nn.Linear(3 * hidden_dim, hidden_dim, bias=False)
        self.student_projection = nn.Linear(3 * hidden_dim, hidden_dim, bias=False)

    def contrastive(self, student, teacher, labels):
        batch = labels.numel()
        zero = sum(x.sum() for x in student.values()) * 0
        if batch < 2:
            return zero
        eye = torch.eye(batch, dtype=torch.bool, device=labels.device)
        positives = labels[:, None].eq(labels[None, :]) & ~eye
        usable = positives.any(-1)
        if not usable.any():
            return zero
        # Six anchor directions: teacher->student and student->teacher for L/A/V.
        losses = []
        detached = {m: x.detach() for m, x in teacher.items()}
        for direction, (anchors, candidates) in enumerate(((detached, student), (student, detached))):
            for index, m in enumerate(MODALITIES):
                anchor = F.normalize(anchors[m], dim=-1)
                other = [n for n in MODALITIES if n != m]
                scales = F.softplus(self.raw_scales[direction, index]).clamp_max(100)
                logits = sum(scales[j] * (anchor @ F.normalize(candidates[n], dim=-1).T)
                             for j, n in enumerate(other))
                logits = logits.masked_fill(eye, -1e9)
                log_probability = F.log_softmax(logits, dim=-1)
                row_loss = -(log_probability * positives).sum(-1) / positives.sum(-1).clamp_min(1)
                losses.append(row_loss[usable].mean())
        return torch.stack(losses).mean()

    def representation(self, student, teacher):
        a = self.student_projection(student)
        b = self.teacher_projection(teacher.detach())
        return (1 - F.cosine_similarity(a, b, dim=-1, eps=1e-8)).mean()

    @staticmethod
    def task(output, targets):
        if output.logits is not None:
            classes = targets.clamp(-3, 3).round().long() + 3
            return F.cross_entropy(output.logits, classes)
        return F.l1_loss(output.score, targets)

    def forward(self, model, student, teacher, targets):
        task = self.task(student, targets)
        regularization = self.qkv_coefficient * model.qkv_penalty()
        csd = self.contrastive(student.modalities, teacher.modalities, (targets.sign() + 1).long())
        srd = self.representation(student.fused, teacher.fused)
        total = task + self.lambda_reg * regularization + self.lambda_csd * csd + self.lambda_srd * srd
        return dict(total=total, task=task, csd=csd, srd=srd, regularization=regularization)
