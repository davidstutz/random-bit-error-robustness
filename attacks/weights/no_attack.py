from .attack import *
import common.torch


class NoAttack(Attack):
    """
    Simple floating point attack on network weights using additive perturbations.
    """

    def __init__(self):
        """
        Constructor.
        """

        super(NoAttack, self).__init__()

    def run(self, model, testset, objective):
        """
        Run attack.

        :param model: model to attack
        :type model: torch.nn.Module
        :param testset: test set
        :type testset: torch.utils.data.DataLoader
        :param objective: objective
        :type objective: UntargetedObjective or TargetedObjective
        """

        super(NoAttack, self).run(model, testset, objective)
        if not self.training:
            return common.torch.clone(model).cpu()
        else:
            return common.torch.clone(model)

