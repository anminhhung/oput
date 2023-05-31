# oput

Optimizer Utils

----
# Updates

| Optimizer   |      Year      |  Reference Code | Paper |
|----------|:-------------:|:------:|------:|
|  [Sophia](#Sophia)  |  2023  |  [Sophia-optim](https://github.com/kyegomez/Sophia) |  [Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://arxiv.org/pdf/2305.14342.pdf)
|  [Momo](#Momo)  |  2023  |  [Momo-optim](https://github.com/fabian-sp/MoMo/tree/main) |  [MoMo: Momentum Models for Adaptive Learning Rates](https://arxiv.org/pdf/2305.07583.pdf)
|  [Lion](#Lion)  |  2023  |  [Lion-optim](https://github.com/google/automl/blob/master/lion/lion_pytorch.py) |  [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/pdf/2302.06675.pdf)
|  [Adan](#Adan)  |  2022  |  [Adan-optim](https://github.com/lucidrains/Adan-pytorch) |  [Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models](https://arxiv.org/pdf/2208.06677.pdf)
|  [MADGRAD](#MADGRAD)  |  2021  | [madgrad](https://github.com/facebookresearch/madgrad) | [Adaptivity without Compromise: A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic Optimization](https://arxiv.org/abs/2101.11075) 
|  [AdaBelief](#AdaBelief)  |  2020  | [AdaBelief-optim](https://github.com/juntang-zhuang/Adabelief-Optimizer) | [AdaBelief Optimizer, adapting stepsizes by the belief in observed gradients](https://arxiv.org/abs/2010.07468) 
|  [AdamP](#AdamP)  |  2020  | [Adamp-optim](https://github.com/clovaai/AdamP) | [Slowing Down the Weight Norm Increase in Momentum-based Optimizers](https://arxiv.org/abs/2006.08217) 
|  [Apollo](#Apollo)  |  2020 | [Apollo-optim](https://github.com/XuezheMax/apollo) |  [Apollo: An Adaptive Parameter-wise Diagonal Quasi-Newton Method for Nonconvex Stochastic Optimization](https://arxiv.org/abs/2009.13586) 
|  [SGDP](#SGDP)  |  2020  | [SGDP-Clovaai](https://github.com/clovaai/AdamP) |  [Slowing Down the Weight Norm Increase in Momentum-based Optimizers](https://arxiv.org/abs/2006.08217) 
|  [AccSGD](#AccSGD)  |  2019  | [AccSGD-optim](https://github.com/rahulkidambi/AccSGD) |  [On the insufficiency of existing momentum schemes for Stochastic Optimization](https://arxiv.org/abs/1803.05591) 
|  [AdaBound](#AdaBound)  |  2019  | [SGD-pytorch-optim](https://github.com/Luolc/AdaBound) |  [Adaptive Gradient Methods with Dynamic Bound of Learning Rate](https://arxiv.org/abs/1902.09843)
|  [AdaMod](#AdaMod)  |  2019  | [AdaBound-optim](https://github.com/lancopku/AdaMod) |  [An Adaptive and Momental Bound Method for Stochastic Learning](https://arxiv.org/abs/1910.12249)
|  [AggMo](#AggMo)  |  2019  | [AdaMod-optim](https://github.com/AtheMathmo/AggMo) |  [Aggregated Momentum: Stability Through Passive Damping](https://arxiv.org/abs/1804.00325)
|  [DiffGrad](#DiffGrad)  |  2019  | [diffGrad-optim](https://github.com/shivram1987/diffGrad) |  [diffGrad: An Optimization Method for Convolutional Neural Networks](https://arxiv.org/abs/1909.11015)
|  [Lamb](#Lamb)  |  2019  | [Lamb-optim](https://github.com/cybertronai/pytorch-lamb) |  [Large Batch Optimization for Deep Learning: Training BERT in 76 minutes](https://arxiv.org/abs/1904.00962)
|  [Lookahead](#Lookahead)  |  2019 | [lookahead-optim](https://github.com/alphadl/lookahead.pytorch)  |  [Lookahead Optimizer: k steps forward, 1 step back](https://arxiv.org/abs/1907.08610)
|  [NovoGrad](#NovoGrad)  |  2019  | [NovoGrad-optim](https://github.com/NVIDIA/DeepLearningExamples/) |  [Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks](https://arxiv.org/abs/1905.11286) 
|  [QHAdam](#QHAdam)  |  2019  | [QHAdam-optim](https://github.com/facebookresearch/qhoptim) |  [Quasi-hyperbolic momentum and Adam for deep learning](https://arxiv.org/abs/1810.06801)
|  [QHM](#QHM)  |  2019   | [qhoptim](https://github.com/facebookresearch/qhoptim) |  [Quasi-hyperbolic momentum and Adam for deep learning](https://arxiv.org/abs/1810.06801)
|  [RAdam](#RAdam)  |  2019  | [RAdam](https://github.com/LiyuanLucasLiu/RAdam)  |  [On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265)
|  [Ranger](#Ranger)  |  2019   | [Ranger-Optimizer](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer) |  [New Deep Learning Optimizer, Ranger: Synergistic combination of RAdam + LookAhead for the best of both](https://medium.com/@lessw/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d)
|  [RangerVA](#RangerVA)  |  2019  | [RangerVA-Optimizer](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer) |  [Calibrating the Adaptive Learning Rate to Improve Convergence of ADAM](https://arxiv.org/abs/1908.00700v2) 
|  [A2GradExp](#A2GradExp)  |  2018   | [A2GradExp-optim](https://github.com/severilov/A2Grad_optimizer) |  [Optimal Adaptive and Accelerated Stochastic Gradient Descent](https://arxiv.org/abs/1810.00553)
|  [A2GradInc](#A2GradInc)  |  2018   | [A2GradInc-optim](https://github.com/severilov/A2Grad_optimizer) |  [Optimal Adaptive and Accelerated Stochastic Gradient Descent](https://arxiv.org/abs/1810.00553)
|  [A2GradUni](#A2GradUni)  |  2018  |  [A2GradUni-optim](https://github.com/severilov/A2Grad_optimizer) | [Optimal Adaptive and Accelerated Stochastic Gradient Descent](https://arxiv.org/abs/1810.00553) 
|  [PID](#PID)  |  2018 | [PIDOptimizer](https://github.com/tensorboy/PIDOptimizer)  |  [A PID Controller Approach for Stochastic Optimization of Deep Networks](http://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR18_PID.pdf) 
|  [RangerQH](#RangerQH)  |  2018  | [RangerQH-Optimizer](https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer) |  [RangerQH *Quasi-hyperbolic momentum and Adam for deep learning](https://arxiv.org/abs/1810.06801)
|  [RangerVA](#Shampoo)  |  2018  | [shampoo.pytorch](https://github.com/moskomule/shampoo.pytorch)  |  [Shampoo: Preconditioned Stochastic Tensor Optimization](https://arxiv.org/abs/1802.09568)
|  [Yogi](#Yogi)  |  2018  | [Yogi-Optimizer](https://github.com/4rtemi5/Yogi-Optimizer_Keras) |  [Adaptive Methods for Nonconvex Optimization](https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization) 
|  [SWATS](#SWATS)  |  2017  | [swats](https://github.com/Mrpatekful/swats)  | [Improving Generalization Performance by Switching from Adam to SGD](https://arxiv.org/abs/1712.07628)
|  [SGDW](#SGDW)  |  2017  | [SGDW-pytorch](https://github.com/pytorch/pytorch/pull/22466) |  [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) 
|  [SGD](#SGD)  |  ...  |  [SGD-pytorch-optim](https://github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py) |  ...
|  [Adam](#Adam)  |  2015  | [Adam-pytorch-optim](https://github.com/pytorch/pytorch/blob/main/torch/optim/adam.py) |  [Adam: A Method for Stochastic Optimization](https://arxiv.org/pdf/1412.6980.pdf)


------
## Sophia

```
import oput
# model = ...
optimizer = oput.SophiaG(
    model.parameters(), 
    lr=2e-4, 
    betas=(0.965, 0.99), 
    rho = 0.01, 
    weight_decay=1e-1
)

optimizer.step()
```

The learning rate is a crucial hyperparameter that controls the step size of the parameter updates during the optimization process. In Decoupled Sophia, the update is written as 
```
p.data.addcdiv_(-group['lr'], m, h.add(group['rho']))
```
which is equivalent to the update in the paper up to a re-parameterization.

## Tips for tuning the learning rate:
Choose the learning rate to be about half the learning rate that you would use for AdamW. Some partial ongoing results indicate that the learning rate can be made even larger, possibly leading to faster convergence. Rho (rho) The rho parameter is used in the update rule to control the Hessian's influence on the parameter updates. It is essential to choose an appropriate value for rho to balance the trade-off between the gradient and the Hessian information.

## Tips for tuning rho:
Consider choosing rho in the range of 0.03 to 0.04. The rho value seems transferable across different model sizes. For example, rho = 0.03 can be used in 125M and 335M Sophia-G models.

The (lr, rho) for 335M Sophia-G is chosen to be (2e-4, 0.03). Though we suspect that the learning rate can be larger, it's essential to experiment with different values to find the best combination for your specific use case.

## Other Hyperparameters
While the learning rate and rho are the most critical hyperparameters to tune, you may also experiment with other hyperparameters such as betas, weight_decay, and k (the frequency of Hessian updates). However, the default values provided in the optimizer should work well for most cases.

Remember that hyperparameter tuning is an iterative process, and the best values may vary depending on the model architecture and dataset. Don't hesitate to experiment with different combinations and validate the performance on a held-out dataset or using cross-validation.

Feel free to share your findings and experiences during hyperparameter tuning. Your valuable feedback and comments can help improve the optimizer and its usage in various scenarios.

## Short-term Goals
Ready to train plug in and play file with your own model or Andromeda

Performance improvements: Investigate and implement potential performance improvements to further reduce training time and computational resources -> Decoupled Sophia + heavy metric logging + Implement in Triton and or Jax?

Additional Hessian estimators: Research and implement other Hessian estimators to provide more options for users.

Hyperparameter tuning: Develop a set of recommended hyperparameters for various use cases and model architectures.

## Mid-term Goals
Integration with Andromeda model: Train the Andromeda model using the Sophia optimizer and compare its performance with other optimizers.

Sophia optimizer variants: Explore and develop variants of the Sophia optimizer tailored for specific tasks, such as computer vision, multi-modality AI, and natural language processing, and reinforcement learning.

Distributed training: Implement support for distributed training to enable users to train large-scale models using Sophia across multiple devices and nodes.

Automatic hyperparameter tuning: Develop an automatic hyperparameter tuning module to help users find the best hyperparameters for their specific use case.

## Long-term Goals
Training multiple models in parallel: Develop a framework for training multiple models concurrently with different optimizers, allowing users to test and compare the performance of various optimizers, including Sophia, on their specific tasks.

Sophia optimizer for other domains: Adapt the Sophia optimizer for other domains, such as optimization in reinforcement learning, Bayesian optimization, and evolutionary algorithms.

By following this roadmap, we aim to make the Sophia optimizer a powerful and versatile tool for the deep learning community, enabling users to train their models more efficiently and effectively.

---------
## Momo

Use Momo optimizer

```
import oput
# model = ...
optimizer = oput.Momo(
    model.parameters(), 
    lr=1e-2
)
```

Use MomoAdam optimizer

```
import oput
# model = ...
optimizer = oput.MomoAdam(
    model.parameters(), 
    lr=1e-2
)
```

**Note that Momo needs access to the value of the batch loss**. In the *.step()* method, you need to pass either

- the loss tensor (when backward has already been done) to the argument *loss*
- or a callable *closure* to the argument *closure* that computes gradients and returns the loss.

For example:
```
def loss_fn(criterion, running_loss, outputs, labels):
    loss = criterion(outputs, labels)
    running_loss += loss.item()
    loss.backward()

    return loss

# in each training step, use:
outputs = model(images)
optimizer.zero_grad()
closure = lambda: loss_fn(criterion, running_loss, outputs, labels) # define a closure that return loss
optimizer.step(closure)
```

---------
## Lion
```
import oput

# model = ...
optimizer = oput.Lion(
    model.parameters(),
    betas=(0.9, 0.99), 
    weight_decay=0.0
)

optimizer.step()
```

---------
## Adan


```
import oput

# model = ...
optimizer = oput.Adan(
    model.parameters(),
    lr = 1e-3,                 
    betas = (0.1, 0.1, 0.001),
    weight_decay = 0       
)

optimizer.step()
```

---------
## SAM

```
import oput

# model = ...
base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
optimizer = oput.SAM(
    model.parameters(), 
    base_optimizer, 
    lr=0.1, 
    momentum=0.9
)

optimizer.step()
```

---------
## A2GradExp


```
import oput

# model = ...
optimizer = oput.A2GradExp(
    model.parameters(),
    beta=10.0,
    lips=10.0,
    rho=0.5,
)

optimizer.step()
```

---------
## A2GradInc

```
import oput

# model = ...
optimizer = oput.A2GradInc(
    model.parameters(),
    beta=10.0,
    lips=10.0,
)

optimizer.step()
```

---------
## A2GradUni


```
import oput

# model = ...
optimizer = oput.A2GradUni(
    model.parameters(),
    beta=10.0,
    lips=10.0,
)

optimizer.step()
```

------
## AccSGD

```
import oput

# model = ...
optimizer = oput.AccSGD(
    model.parameters(),
    lr=1e-3,
    kappa=1000.0,
    xi=10.0,
    small_const=0.7,
    weight_decay=0
)

optimizer.step()
```

---------

## AdaBelief

```
import oput

# model = ...
optimizer = oput.AdaBelief(
    model.parameters(),
    lr= 1e-3,
    betas=(0.9, 0.999),
    eps=1e-3,
    weight_decay=0,
    amsgrad=False,
    weight_decouple=False,
    fixed_decay=False,
    rectify=False,
)

optimizer.step()
```

--------
## AdaBound

```
import oput

# model = ...
optimizer = oput.AdaBound(
    model.parameters(),
    lr= 1e-3,
    betas= (0.9, 0.999),
    final_lr = 0.1,
    gamma=1e-3,
    eps= 1e-8,
    weight_decay=0,
    amsbound=False,
)

optimizer.step()
```


------
## AdaMod

```
import oput

# model = ...
optimizer = oput.AdaMod(
    model.parameters(),
    lr= 1e-3,
    betas=(0.9, 0.999),
    beta3=0.999,
    eps=1e-8,
    weight_decay=0,
)

optimizer.step()
```

------
## AdamP

```
import oput

# model = ...
optimizer = oput.AdamP(
    model.parameters(),
    lr= 1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
    delta = 0.1,
    wd_ratio = 0.1
)

optimizer.step()
```

-----
## AggMo

```
import oput

# model = ...
optimizer = oput.AggMo(
    model.parameters(),
    lr= 1e-3,
    betas=(0.0, 0.9, 0.99),
    weight_decay=0,
)

optimizer.step()
```

------
## Apollo

```
import oput

# model = ...
optimizer = oput.Apollo(
    model.parameters(),
    lr= 1e-2,
    beta=0.9,
    eps=1e-4,
    warmup=0,
    init_lr=0.01,
    weight_decay=0,
)

optimizer.step()
```

---------
## DiffGrad


```
import oput

# model = ...
optimizer = oput.DiffGrad(
    model.parameters(),
    lr= 1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
)

optimizer.step()
```


---------
## Lamb

```
import oput

# model = ...
optimizer = oput.Lamb(
    model.parameters(),
    lr= 1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
)

optimizer.step()
```

---------
## Lookahead

```
import oput

# model = ...
# base optimizer, any other optimizer can be used like Adam or DiffGrad
yogi = oput.Yogi(
    model.parameters(),
    lr= 1e-2,
    betas=(0.9, 0.999),
    eps=1e-3,
    initial_accumulator=1e-6,
    weight_decay=0,
)

optimizer = oput.Lookahead(yogi, k=5, alpha=0.5)
optimizer.step()
```

---------
## MADGRAD

```
import oput

# model = ...
optimizer = oput.MADGRAD(
    model.parameters(),
    lr=1e-2,
    momentum=0.9,
    weight_decay=0,
    eps=1e-6,
)

optimizer.step()
```

--------
## NovoGrad

```
import oput

# model = ...
optimizer = oput.NovoGrad(
    model.parameters(),
    lr= 1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
    grad_averaging=False,
    amsgrad=False,
)

optimizer.step()
```

--------
## PID

```
import oput

# model = ...
optimizer = oput.PID(
    model.parameters(),
    lr=1e-3,
    momentum=0,
    dampening=0,
    weight_decay=1e-2,
    integral=5.0,
    derivative=10.0,
)

optimizer.step()
```


------
## QHAdam

```
import oput

# model = ...
optimizer = oput.QHAdam(
    m.parameters(),
    lr= 1e-3,
    betas=(0.9, 0.999),
    nus=(1.0, 1.0),
    weight_decay=0,
    decouple_weight_decay=False,
    eps=1e-8,
)

optimizer.step()
```

------
## QHM

```
import oput

# model = ...
optimizer = oput.QHM(
    model.parameters(),
    lr=1e-3,
    momentum=0,
    nu=0.7,
    weight_decay=1e-2,
    weight_decay_type='grad',
)

optimizer.step()
```

-----
## RAdam

```
import oput

# model = ...
optimizer = oput.RAdam(
    model.parameters(),
    lr= 1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
)

optimizer.step()
```


------
## Ranger

```
import oput

# model = ...
optimizer = oput.Ranger(
    model.parameters(),
    lr=1e-3,
    alpha=0.5,
    k=6,
    N_sma_threshhold=5,
    betas=(.95, 0.999),
    eps=1e-5,
    weight_decay=0
)

optimizer.step()
```

--------
## RangerQH

```
import oput

# model = ...
optimizer = oput.RangerQH(
    model.parameters(),
    lr=1e-3,
    betas=(0.9, 0.999),
    nus=(.7, 1.0),
    weight_decay=0.0,
    k=6,
    alpha=.5,
    decouple_weight_decay=False,
    eps=1e-8,
)

optimizer.step()
```


--------
## RangerVA

```
import oput

# model = ...
optimizer = oput.RangerVA(
    model.parameters(),
    lr=1e-3,
    alpha=0.5,
    k=6,
    n_sma_threshhold=5,
    betas=(.95, 0.999),
    eps=1e-5,
    weight_decay=0,
    amsgrad=True,
    transformer='softplus',
    smooth=50,
    grad_transformer='square'
)

optimizer.step()
```

----
## SGDP

```
import oput

# model = ...
optimizer = oput.SGDP(
    model.parameters(),
    lr= 1e-3,
    momentum=0,
    dampening=0,
    weight_decay=1e-2,
    nesterov=False,
    delta = 0.1,
    wd_ratio = 0.1
)

optimizer.step()
```

----
## SGDW

```
import oput

# model = ...
optimizer = oput.SGDW(
    model.parameters(),
    lr= 1e-3,
    momentum=0,
    dampening=0,
    weight_decay=1e-2,
    nesterov=False,
)

optimizer.step()
```

-----
## SWATS

```
import oput

# model = ...
optimizer = oput.SWATS(
    model.parameters(),
    lr=1e-1,
    betas=(0.9, 0.999),
    eps=1e-3,
    weight_decay= 0.0,
    amsgrad=False,
    nesterov=False,
)

optimizer.step()
```

-------
## Shampoo

```
import oput

# model = ...
optimizer = oput.Shampoo(
    model.parameters(),
    lr=1e-1,
    momentum=0.0,
    weight_decay=0.0,
    epsilon=1e-4,
    update_freq=1,
)

optimizer.step()
```


----
## Yogi

```
import optim

# model = ...
optimizer = optim.Yogi(
    model.parameters(),
    lr= 1e-2,
    betas=(0.9, 0.999),
    eps=1e-3,
    initial_accumulator=1e-6,
    weight_decay=0,
)

optimizer.step()
```

-----------------------
## Adam 

```
import oput

# model = ...
optimizer = oput.Adam(
    model.parameters(), 
    lr=0.0001,
    betas=(0.9, 0.999), 
    eps=1e-8,
    weight_decay=0
)

optimizer.step()
```

----------------------
## SGD

```
import oput

# model = ...
optimizer = oput.SGD(
    model.parameters(), 
    lr=0.0001,
    momentum=0, 
    dampening=0,
    weight_decay=0
)

optimizer.step()
```