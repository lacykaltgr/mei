# Most Exciting Input
Generate ***Most Exciting Input*** to explore and understand PyTorch model's behavior by identifying input samples that induce high activation from specific neurons in your model.

## Paper
paper: [Input optimization for interpreting neural generative models](https://lacykaltgr.github.io/assets/pdf/TDK2023.pdf)

## Installation
```bash
pip install meitorch
```

## Usage

1. Load the model you want to generate interpretable visualizations for.

```python
model = load_your_model()
```

2. Define the operation you want to optimize most exciting inputs for.

Your operation must take a batch of inputs as parameters and return a dictionary of losses. 
Visualizations will be generated for the input that maximizes the loss named "objective".


```python
import torch

def operation(inputs):
    outputs = model(inputs)
    activation = outputs[:, 0]
    activation = torch.mean(activation, dim=0)
    loss = -activation
    losses = dict(
        objective=loss,
        activation=activation,
    )
    return losses
```

You can also define more complex operations that include multiple losses. 
Adding other losses to the dictionary will enable you to plot them after the optimization.

```python   
def operation(inputs):
    outputs = model(inputs)
    activation = outputs[:, 0]
    activation = torch.mean(activation, dim=0)
    model_losses = compute_loss(inputs, outputs)
    regularization = losses["elbo"] * 0.1
    loss = -activation + regularization
    losses = dict(
        objective=loss,
        activation=activation,
        elbo_regularization=regularization,
    )
   return losses
```

3. Create a MEI object with your operation and the input shape of your model.

```python
from meitorch.mei import MEI

mei = MEI(operation=operation(), shape=(1, 40, 40), device=device)
```

4. Define a configuration for the optimization and generate **Most Exciting Inputs**.
There are minor differences between the configuration for different optimization schemes, more on configurations below.

**Generate pixel-wise MEI**
```python
pixel_mei_config = dict(/* your config here */)
result = mei.generate_pixel_mei(config=pixel_mei_config)
```
**Generate variational MEI**
```python
variational_mei_config = dict(/* your config here */)
result = mei.generate_variational_mei(config=variational_mei_config)
```

**Generate transformation MEI**
```python
transformation_mei_config = dict(/* your config here */)
result = mei.generate_transformation_mei(config=transformation_mei_config)
```

5. Analyze the results
Access the generated images and the losses from the result object.

**Plot the loss curves and the visualizations**
```python
result.plot_losses(show=False, save_path=None, ranges=None)
result.plot_image_and_losses(self, save_path=None, ranges=None)
```

**Plot spatial frequency spectrum of the generated images**
```python
result.plot_spatial_frequency_spectrum()
```
**Further analysis**
You can further analyze the results with the **meitorch.analyze** module.
```python
from meitorch.analyze import Analyze
```

## Configurations

For all configurations, you can use a schedule instead of a constant value for any parameter.
A schedule is a function that takes the current iteration as input and returns the value for that iteration.
You can access the schedule class in **meitorch.tools.schedules**.
```python
from meitorch.tools.schedules import LinearSchedule

schedule = LinearSchedule(start=0.1, end=0.01)
```
Available schedules are:
```
- LinearSchedule(start, end)
- OctaveSchedule(values)
- RandomSchedule(minimum, maximum)
```

**Pixel-wise MEI configuration example**

```python
image_mei_config = dict(
    iter_n=2,         # number of optimization steps
    n_samples=1,      # number of samples per batch
    save_every=1,     # save copy of image every n iterations
    bias=0,           # bias of the distribution the image is sampled from
    scale=1,          # scaling of the distribution the image is sampled from
    diverse=False,    # whether to use diverse sampling
    diverse_params=dict(
        div_metric='euclidean', # distance metric for diversity (euclidean, cosine, correlation)
        div_linkage='minimum',  # linkage criterion for diversity (minimum, average)
        div_weight=1.1,         # weight of diversity loss
    ),

    #pre-step transformations
    scaler=1.01,          # scaling of the image before each step
    jitter=3,             # size of translational jittering before each step

    #normalization/clipping
    train_norm=1,        # norm adjustment during step
    norm=1,              # norm adjustment after step

    #optmizer
    optimizer="rmsprop",    # optimizer (sgd, mei, rmsprop, adam)
    optimizer_params=dict(
        lr=0.03,            # learning rate
        weight_decay=1e-6,  # weight decay
    ),

    #preconditioning in the gradient
    precond=0.3,            # strength of gradient preconditioning filter falloff (float or schedule)

    #denoiser after each step
    blur='gaussian',        # denoiser type (gaussian, tv, bilateral)
    blur_params=dict(
        #gaussian
        kernel_size=3,
        sigma=LinearSchedule(0.1, 0.01)
        
        #tv
        #regularization_scaler=1e-7,
        #lr=0.0001,
        #num_iters=5,
        
        #bilateral
        #kernel_size=3,
        #sigma_color=LinearSchedule(1, 0.01),
        #sigma_spatial=LinearSchedule(0.25, 0.01),
    ),
)
```

**Variational MEI configuration example**

```python
var_mei_config = dict(
    iter_n=1,              # number of optimization steps
    save_every=100,        # save image every n iterations
    bias=0,                # bias of the distribution the image is sampled from
    scale=1,              # scaling of the distribution the image is sampled from

    #transformations
    scaler=RandomSchedule(1, 1.025),  # scaling of the image (float or schedule)
    jitter=None,                      # size of translational jittering

    #optmizer
    optimizer="rmsprop",        # optimizer (sgd, mei, rmsprop, adam)
    optimizer_params=dict(   
        lr=0.04,                # learning rate
        weight_decay=1e-7,      # weight decay
    ),

    #preconditioning
    precond=0.4,            # strength of gradient preconditioning filter

    #variational
    distribution='normal',      # distribution of the MEI (normal, laplace)
    n_samples_per_batch=(128,), # number of samples per batch (tuple)
    fixed_stddev=0.4,           # fixed stddev of the distribution, None for learned stddev
)
```

**Transformation MEI configuration example**

For the transformation MEI, you need to define a transformation operation that takes an image as input and returns a transformed image. 
Any backpropagatable operation can be used as a transformation. In the example below, we use a generative convolutional network, which is defined in **meitorch.tools.transformations**.

```python
tranformation_mei_config = dict(
        iter_n=150,          # number of optimization steps
        save_every=1,        # save image every n iterations
        bias=0,              # bias of the distribution the image is sampled from
        scale=1,             # scaling of the distribution the image is sampled from
        n_samples=128,       # number of samples per batch

        #transformations before each step
        scaler=None,            # scaling of the image (float or schedule)
        jitter=None,            # size of translational jittering

        #normalization
        train_norm=None,        # norm adjustment during step

        #optmizer
        optimizer="mei",        # optimizer (sgd, mei, rmsprop, adam)
        optimizer_params=dict
        (
            lr=0.02,            # learning rate
            weight_decay=1e-5,  # weight decay
        ),
    
        #preconditioning
        precond=0.4,            # strength of gradient preconditioning filter

        # transformation operation
        transform = GenerativeConvNet(hidden_sizes=[1], fixed_stddev=0.6, kernel_size=9,  activation=torch.nn.ReLU(), activate_output=False, shape=(1, 40, 40))
    )
```