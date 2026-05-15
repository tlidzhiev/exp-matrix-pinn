import torch
import torch.nn.functional as F
from torchmetrics import Metric

from src.metrics.base import BaseMetric


class _CosineSimilarityMetric(Metric):
    def __init__(self) -> None:
        super().__init__()
        self.add_state('sum_cos_sim', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total_samples', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        B, _, _ = target.shape
        cos_sim = F.cosine_similarity(target, preds, dim=-1)
        self.sum_cos_sim += cos_sim.mean(dim=1).sum()
        self.total_samples += B

    def compute(self) -> torch.Tensor:
        return self.sum_cos_sim / self.total_samples


class CosineSimilarityMetric(BaseMetric):
    def __init__(self, name: str | None = None, device: str = 'auto', **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.metric = _CosineSimilarityMetric().to(device)

    def update(
        self,
        ut_pred: torch.Tensor,
        ut: torch.Tensor,
        **kwargs,
    ) -> None:  # ty:ignore[invalid-method-override]
        with torch.no_grad():
            self.metric.update(ut_pred, ut)  # ty:ignore[invalid-argument-type]

    def __call__(
        self,
        ut_pred: torch.Tensor | None = None,
        ut: torch.Tensor | None = None,
        **kwargs,
    ) -> float:
        if ut_pred is not None and ut is not None:
            self.update(ut_pred=ut_pred, ut=ut)
        result = self.metric.compute()  # ty:ignore[missing-argument]
        self.metric.reset()
        return result.item()
