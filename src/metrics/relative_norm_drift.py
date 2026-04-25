import torch
from torchmetrics import Metric

from src.metrics.base import BaseMetric


class _RelativeNormDriftMetric(Metric):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.add_state('sum_drift', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total_samples', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, u0: torch.Tensor) -> None:
        B, _ = u0.shape
        pred_norm = torch.linalg.vector_norm(preds, dim=-1)
        u0_norm = torch.linalg.vector_norm(u0, dim=-1, keepdim=True)
        drift = torch.abs(pred_norm - u0_norm) / (u0_norm + self.eps)
        self.sum_drift += drift.mean(dim=1).sum()
        self.total_samples += B  # ty:ignore[unsupported-operator]

    def compute(self) -> torch.Tensor:
        return self.sum_drift / self.total_samples  # ty:ignore[unsupported-operator]


class RelativeNormDriftMetric(BaseMetric):
    def __init__(
        self, name: str | None = None, device: str = 'auto', eps: float = 1e-8, **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.metric = _RelativeNormDriftMetric(eps=eps).to(device)

    def update(
        self,
        ut_pred: torch.Tensor,
        u0: torch.Tensor,
        **kwargs,
    ) -> None:  # ty:ignore[invalid-method-override]
        with torch.no_grad():
            if ut_pred.dim() == 2:
                ut_pred = ut_pred.unsqueeze(1)
            self.metric.update(ut_pred, u0)  # ty:ignore[invalid-argument-type]

    def __call__(
        self,
        ut_pred: torch.Tensor | None = None,
        u0: torch.Tensor | None = None,
        **kwargs,
    ) -> float:
        if ut_pred is not None and u0 is not None:
            self.update(ut_pred=ut_pred, u0=u0)
        result = self.metric.compute()  # ty:ignore[missing-argument]
        self.metric.reset()
        return result.item()
