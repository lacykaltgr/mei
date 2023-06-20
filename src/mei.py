import torch
import numpy as np
from gabor import Gabor
from utils import adj_model, contrast_tuning
from mei_process import MEIProcess


class MEI(Gabor):
    def __init__(self, models, shape=(1, 28, 28), bias=0, scale=1, device='cpu'):
        super().__init__(models, shape, bias, scale, device)


    #TODO: implement WRONGMEI : generálásnál a mean-t használja std helyett is
    def generate(self, neuron_query, **MEIParams):

        process = MEIProcess(adj_model(self.models, neuron_query), bias=self.bias, scale=self.scale, device=self.device, **MEIParams)

        # generate initial random image
        background_color = np.float32([self.bias] * min(self.img_shape))
        gen_image = np.random.normal(background_color, self.scale / 20, self.img_shape)
        gen_image = np.clip(gen_image, -1, 1)

        # generate class visualization via octavewise gradient ascent
        gen_image = process.deepdraw(gen_image, random_crop=False)
        mei = gen_image.squeeze()

        with torch.no_grad():
            img = torch.Tensor(gen_image[None, ...]).to(self.device)
            activation = process.operation(img).data.cpu().numpy()[0]

        #cont, vals, lim_contrast = contrast_tuning(adj_model, mei, device=self.device)

        process.mei = mei
        process.activation = activation
        #process.monotonic = bool(np.all(np.diff(vals) >= 0))
        #process.max_activation = np.max(vals)
        #process.max_contrast = cont[np.argmax(vals)]
        #process.sat_contrast = np.max(cont)
        #process.img_mean = mei.mean()
        #process.lim_contrast = lim_contrast
        return process

    def gradient_rf(self, neuron_query, **MEIParams):
        def init_rf_image(stimulus_shape=(1, 36, 64)):
            return torch.zeros(1, *stimulus_shape, device='cuda', requires_grad=True)

        def linear_model(x):
            return (x * rf).sum()

        process = MEIProcess(adj_model(linear_model, neuron_query), bias=self.bias, scale=self.scale, device=self.device, **MEIParams)


        X = init_rf_image(self.img_shape[1:])
        y = process.operation(X)
        y.backward()
        point_rf = X.grad.data.cpu().numpy().squeeze()
        rf = X.grad.data

        # generate initial random image
        background_color = np.float32([self.bias] * min(self.img_shape))
        gen_image = np.random.normal(background_color, self.scale / 20, self.img_shape)
        gen_image = np.clip(gen_image, -1, 1)

        # generate class visualization via octavewise gradient ascent
        gen_image = process.deepdraw(gen_image, random_crop=False)
        rf = gen_image.squeeze()

        with torch.no_grad():
            img = torch.Tensor(gen_image[None, ...]).to(self.device)
            activation = process.operation(img).data.cpu().numpy()[0]

        cont, vals, lim_contrast = contrast_tuning(adj_model, rf, self.bias, self.scale)
        process.mei = rf
        process.monotonic = bool(np.all(np.diff(vals) >= 0))
        process.max_activation = np.max(vals)
        process.max_contrast = cont[np.argmax(vals)]
        process.sat_contrast = np.max(cont)
        process.point_rf = point_rf
        process.activation = activation
        return process





