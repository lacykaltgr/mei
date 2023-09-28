import torch

from .linearmei import LinearMEI
from result import MEI_image, MEI_distibution, MEI_neural_network
from tools.deepdraw import deepdraw
from tools.generator_nn import get_net


class MEI(LinearMEI):
    """
    Class for generating more complex optimized inputs
    """
    def __init__(self, operations, shape=(1, 28, 28), device='cpu'):
        super().__init__(operations, shape, device)

    def generate_image_based(self, **MEIParams):
        """
        Generate most exciting inputs
        Uses deepdraw to optimize images
        :param neuron_query: The queried neurons of the output layer.
        :param MEIParams: Additional parameters for the optimization process.
        :return: Process(es) with MEI images
        """
        processes = []
        n_samples = MEIParams["n_samples"]
        for op in self.operations:
            if MEIParams["gradient_rf"]:
                op, pointrf = self.to_gradient_rf_op(op)
            process = MEI_image(n_samples, self.img_shape, **MEIParams)
            result_stats = deepdraw(process, op)
            process.result_dict.update(result_stats)
            processes.append(process)
        return processes if len(processes) > 1 else processes[0]

    def generate_variational(self, **MEIParams):
        processes = []
        distribution = MEIParams["distribution"]
        for op in self.operations:
            if MEIParams["gradient_rf"]:
                op, pointrf = self.to_gradient_rf_op(op)
            process = MEI_distibution(distribution, self.img_shape, **MEIParams)
            result_dict = deepdraw(process, op)
            process.result_dict.update(result_dict)
            processes.append(process)
        return processes if len(processes) > 1 else processes[0]

    def generate_nn_based(self, **MEIParams):
        processes = []
        input_type = MEIParams["input_type"]
        input_shape = MEIParams["input_shape"]
        for op in self.operations:
            if MEIParams["gradient_rf"]:
                op, pointrf = self.to_gradient_rf_op(op)
            net = get_net(MEIParams["net"])
            process = MEI_neural_network(net,input_type, input_shape, self.img_shape, **MEIParams)
            result_dict = deepdraw(process, op)
            process.result_dict.update(result_dict)
            processes.append(process)
        return processes if len(processes) > 1 else processes[0]

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










