# oput

Optimizer Utils


---------
A2GradExp


```
import oput

# model = ...
optimizer = oput.A2GradExp(
    model.parameters(),
    beta=10.0,
    lips=10.0,
    rho=0.5,
)
```

**Paper**: *Optimal Adaptive and Accelerated Stochastic Gradient Descent* (2018) [https://arxiv.org/abs/1810.00553]

**Reference Code**: https://github.com/severilov/A2Grad_optimizer

---------
A2GradInc

```
import oput

# model = ...
optimizer = oput.A2GradInc(
    model.parameters(),
    beta=10.0,
    lips=10.0,
)
```

**Paper**: *Optimal Adaptive and Accelerated Stochastic Gradient Descent* (2018) [https://arxiv.org/abs/1810.00553]

**Reference Code**: https://github.com/severilov/A2Grad_optimizer

---------
A2GradUni


```
import oput

# model = ...
optimizer = oput.A2GradUni(
    model.parameters(),
    beta=10.0,
    lips=10.0,
)
```

**Paper**: *Optimal Adaptive and Accelerated Stochastic Gradient Descent* (2018) [https://arxiv.org/abs/1810.00553]

**Reference Code**: https://github.com/severilov/A2Grad_optimizer

------
AccSGD

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
```

**Paper**: *On the insufficiency of existing momentum schemes for Stochastic Optimization* (2019) [https://arxiv.org/abs/1803.05591]

**Reference Code**: https://github.com/rahulkidambi/AccSGD

---------

AdaBelief

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
```

**Paper**: *AdaBelief Optimizer, adapting stepsizes by the belief in observed gradients* (2020) [https://arxiv.org/abs/2010.07468]

**Reference Code**: https://github.com/juntang-zhuang/Adabelief-Optimizer

--------
AdaBound

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
```

**Paper**: *Adaptive Gradient Methods with Dynamic Bound of Learning Rate* (2019) [https://arxiv.org/abs/1902.09843]

**Reference Code**: https://github.com/Luolc/AdaBound

------
AdaMod

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
```

**Paper**: *An Adaptive and Momental Bound Method for Stochastic Learning.* (2019) [https://arxiv.org/abs/1910.12249]

**Reference Code**: https://github.com/lancopku/AdaMod

------
AdamP

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

**Paper**: *Slowing Down the Weight Norm Increase in Momentum-based Optimizers.* (2020) [https://arxiv.org/abs/2006.08217]

**Reference Code**: https://github.com/clovaai/AdamP

-----
AggMo

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

**Paper**: *Aggregated Momentum: Stability Through Passive Damping.* (2019) [https://arxiv.org/abs/1804.00325]

**Reference Code**: https://github.com/AtheMathmo/AggMo

------
Apollo

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

**Paper**: *Apollo: An Adaptive Parameter-wise Diagonal Quasi-Newton Method for Nonconvex Stochastic Optimization.* (2020) [https://arxiv.org/abs/2009.13586]

**Reference Code**: https://github.com/XuezheMax/apollo

---------
DiffGrad


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

**Paper**: *diffGrad: An Optimization Method for Convolutional Neural Networks.* (2019) [https://arxiv.org/abs/1909.11015]

**Reference Code**: https://github.com/shivram1987/diffGrad

---------
Lamb

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

**Paper**: *Large Batch Optimization for Deep Learning: Training BERT in 76 minutes* (2019) [https://arxiv.org/abs/1904.00962]

**Reference Code**: https://github.com/cybertronai/pytorch-lamb

---------
Lookahead

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

**Paper**: *Lookahead Optimizer: k steps forward, 1 step back* (2019) [https://arxiv.org/abs/1907.08610]

**Reference Code**: https://github.com/alphadl/lookahead.pytorch

---------
MADGRAD

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

**Paper**: *Adaptivity without Compromise: A Momentumized, Adaptive, Dual Averaged Gradient Method for Stochastic Optimization* (2021) [https://arxiv.org/abs/2101.11075]

**Reference Code**: https://github.com/facebookresearch/madgrad

--------
NovoGrad

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

**Paper**: *Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks* (2019) [https://arxiv.org/abs/1905.11286]

**Reference Code**: https://github.com/NVIDIA/DeepLearningExamples/

--------
PID

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

**Paper**: *A PID Controller Approach for Stochastic Optimization of Deep Networks* (2018) [http://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR18_PID.pdf]

**Reference Code**: https://github.com/tensorboy/PIDOptimizer

------
QHAdam

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

**Paper**: *Quasi-hyperbolic momentum and Adam for deep learning* (2019) [https://arxiv.org/abs/1810.06801]

**Reference Code**: https://github.com/facebookresearch/qhoptim

------
QHM

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

**Paper**: *Quasi-hyperbolic momentum and Adam for deep learning* (2019) [https://arxiv.org/abs/1810.06801]

**Reference Code**: https://github.com/facebookresearch/qhoptim

-----
RAdam

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

**Paper**: *On the Variance of the Adaptive Learning Rate and Beyond* (2019) [https://arxiv.org/abs/1908.03265]

**Reference Code**: https://github.com/LiyuanLucasLiu/RAdam

------
Ranger

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

**Paper**: *New Deep Learning Optimizer, Ranger: Synergistic combination of RAdam + LookAhead for the best of both* (2019) [https://medium.com/@lessw/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d]

**Reference Code**: https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer

--------
RangerQH

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

**Paper**: *Quasi-hyperbolic momentum and Adam for deep learning* (2018) [https://arxiv.org/abs/1810.06801]

**Reference Code**: https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer

--------
RangerVA

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

**Paper**: *Calibrating the Adaptive Learning Rate to Improve Convergence of ADAM* (2019) [https://arxiv.org/abs/1908.00700v2]

**Reference Code**: https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer

----
SGDP

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

**Paper**: *Slowing Down the Weight Norm Increase in Momentum-based Optimizers.* (2020) [https://arxiv.org/abs/2006.08217]

**Reference Code**: https://github.com/clovaai/AdamP

----
SGDW

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

**Paper**: *SGDR: Stochastic Gradient Descent with Warm Restarts* (2017) [https://arxiv.org/abs/1608.03983]

**Reference Code**: https://github.com/pytorch/pytorch/pull/22466

-----
SWATS

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

**Paper**: *Improving Generalization Performance by Switching from Adam to SGD* (2017) [https://arxiv.org/abs/1712.07628]

**Reference Code**: https://github.com/Mrpatekful/swats

-------
Shampoo

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

**Paper**: *Shampoo: Preconditioned Stochastic Tensor Optimization* (2018) [https://arxiv.org/abs/1802.09568]

**Reference Code**: https://github.com/moskomule/shampoo.pytorch

----
Yogi

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

**Paper**: *Adaptive Methods for Nonconvex Optimization* (2018) [https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization]

**Reference Code**: https://github.com/4rtemi5/Yogi-Optimizer_Keras

-----------------------
Adam 

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

**Paper**: *Adam: A Method for Stochastic Optimization* (2015) [https://arxiv.org/pdf/1412.6980.pdf]

**Reference Code**: https://github.com/pytorch/pytorch/blob/main/torch/optim/adam.py

----------------------
SGD

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
**Reference Code**: https://github.com/pytorch/pytorch/blob/main/torch/optim/sgd.py