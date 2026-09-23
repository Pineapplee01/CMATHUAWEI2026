import torch

from aumdf.model import AUMDF
from aumdf.losses import DistillationLoss, freeze_teacher
from test_model import inputs, small_config


def test_teacher_frozen_student_and_distillation_projections_trainable():
    teacher, student = AUMDF(small_config()), AUMDF(small_config())
    freeze_teacher(teacher)
    criterion = DistillationLoss(hidden_dim=12)
    xs, masks = inputs(batch=3)
    y = torch.tensor([-1.0, 1.0, 1.5])
    with torch.no_grad():
        target = teacher(xs, masks)
    prediction = student(xs, masks)
    losses = criterion(student, prediction, target, y)
    assert set(losses) == {"total", "task", "csd", "srd", "regularization"}
    assert all(torch.isfinite(v) for v in losses.values())
    losses["total"].backward()
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    assert any(p.grad is not None for p in student.parameters())
    assert criterion.student_projection.weight.grad is not None
    assert criterion.raw_scales.grad is not None


def test_csd_no_positive_pairs_returns_connected_zero():
    criterion = DistillationLoss(hidden_dim=4)
    student = {m: torch.randn(2, 4, requires_grad=True) for m in ("text", "audio", "vision")}
    teacher = {m: x.detach() for m, x in student.items()}
    loss = criterion.contrastive(student, teacher, torch.tensor([0, 2]))
    assert loss.item() == 0
    loss.backward()
    assert student["text"].grad is not None


def test_cosine_distillation_aligned_beats_opposite():
    criterion = DistillationLoss(hidden_dim=4)
    with torch.no_grad():
        criterion.teacher_projection.weight.copy_(torch.eye(4, 12))
        criterion.student_projection.weight.copy_(torch.eye(4, 12))
    x = torch.ones(3, 12)
    assert criterion.representation(x, x) < criterion.representation(-x, x)
