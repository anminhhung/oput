# oput

Optimizer Utils

----
# Updates

| Optimizer   |      Year      |  Reference Code | Paper |
|----------|:-------------:|:------:|------:|
|  [Sgem](#Sgem)  |  2022  |  [Sgem-optim](https://github.com/txping/SGEM/blob/main/optims/sgem.py) |  [SGEM: STOCHASTIC GRADIENT WITH ENERGY AND MOMENTUM](https://arxiv.org/pdf/2208.02208v1.pdf)
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


---------
## Sgem


```
import oput

# model = ...
optimizer = oput.SGEM(
    model.parameters(),
    lr = 1e-3,                 
    c=1.0, 
    momentum=0.9, 
    weight_decay=0   
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
    lr=0.0001
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
    lr=0.0001
    momentum=0, 
    dampening=0,
    weight_decay=0
)

optimizer.step()
```