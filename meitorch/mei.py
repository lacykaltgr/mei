import torch

from .linearmei import LinearMEI
from .result import MEI_image, MEI_distribution, MEI_neural_network
from .objective.deepdraw import deepdraw


class MEI(LinearMEI):
    """
    Class for generating more complex optimized inputs
    """
    def __init__(self, operation, shape=(1, 28, 28),  device='cpu'):
        super().__init__(operation, shape, device=device)

    def generate_image_based(self, init=None, **MEIParams):
        """
        Generate most exciting inputs
        Uses deepdraw to optimize images
        :param init: the initial image (no random generation)
        :param neuron_query: The queried neurons of the output layer.
        :param MEIParams: Additional parameters for the optimization process.
        :return: Process(es) with MEI images
        """
        n_samples = MEIParams["n_samples"]
        del MEIParams["n_samples"]
        process = MEI_image(self.img_shape, n_samples, init=init, device=self.device, **MEIParams)
        return self._generate(process)

    def generate_variational(self, init=None, **MEIParams):
        distribution = MEIParams["distribution"]
        del MEIParams["distribution"]
        process = MEI_distribution(distribution, self.img_shape, init=init, device=self.device, **MEIParams)
        return self._generate(process)

    def generate_nn_based(self, **MEIParams):
        net = MEIParams["net"]
        del MEIParams["net"]
        process = MEI_neural_network(net, self.img_shape, device=self.device, **MEIParams)
        return self._generate(process)

    def _generate(self, process):
        if self.is_gradient_rf_op(process.param_dict):
            op, pointrf = self.to_gradient_rf_op(self.operation)
            process.result_dict.update({"pointrf": pointrf})
        else:
            op = self.operation
        result_dict = deepdraw(process, op)
        process.result_dict.update(result_dict)
        return process

    def is_gradient_rf_op(self, MEIParams):
        return "gradient_rf" in MEIParams and MEIParams["gradient_rf"]

    def to_gradient_rf_op(self, operation):
        """
        Generate most exciting inputs based on the linear function of the gradients of the input
        Uses deepdraw to optimize images
        :return: Process(es) with GradientRF images
        """

        X = torch.zeros(1, *self.img_shape, device=self.device, requires_grad=True)
        y = operation(X)
        y.backward()
        point_rf = X.grad.data.cpu().numpy().squeeze()
        rf = X.grad.data

        def linear_model(x):
            return (x * rf).sum()
        return linear_model, point_rf










