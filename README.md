# CRCD

## Introduction
* This repo is the incomplete implementation  of the following paper:

    "Complementary Relation Contrastive Distillation" (CRCD). [arxiv](https://arxiv.org/abs/2103.16367)

* I am sorry that the source code in this repository is not an official  implementation, which is relies on some internal code of the company's self-developed deep learning library.
However, I reimplented the most critical parts in the works with torch, thus it should be very easy to be pluged into the CRD repo.
I will open-source the complete version later.

## Key components

* Contrastive loss

> the crd-style implementation is [here](./crcd)

* Computation of gradient element
    
> gradient element is computed in the `def get_grad_loss()` in the [loops.py](./helper/loops.py) by using torch API `torch.autograd.grad()`.
    Then, the crcd loss by using gradient relation can be obtained easily.

* the very effective trick
> It is very effective to adjust the distillation loss weight dynamically during the training procedures. We supply some strategy examples in the funtion `def adjust_mimic_loss_weight()` in the [loops.py](./helper/loops.py).
In these strategy, the reregulatization term in the total loss from distillaltion loss is reduced according to a certain rule as the training progresses.

>  In our exps with 250 epochs training, we adopted the stepwise one: before 240 epoch, the loss weight maintains 1; after 240, the loss weight is adjusted to 0 for the last 10 epochs. This means the students are finetuned for another 10 epochs with the minimun learning rate




