"""观测上下文融合仍须满足遮挡无泄漏、账本和补全梯度约束。"""
from tests import test_model_contracts as contracts


class ContextFusionContracts(contracts.ModelContracts):
    def setUp(self):
        super().setUp()
        self.model.cfg['contextual_fusion'] = True
        self.model.cfg['fusion_strength'] = .5
