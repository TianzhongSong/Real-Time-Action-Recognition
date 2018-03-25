"""
SGD schdules during training
"""
from .callbacks import Step


def onetenth_4_8_12(lr):
    steps = [10, 25, 40]
    lrs = [lr, lr / 10, lr / 100, lr / 1000]
    return Step(steps, lrs)

def onetenth_10_20_30(lr):
    steps = [15, 25, 35]
    lrs = [lr, lr / 10, lr / 100, lr / 1000]
    return Step(steps, lrs)

def onetenth_5_10_15(lr):
    steps = [5, 10, 15]
    lrs = [lr, lr / 10, lr / 100, lr / 1000]
    return Step(steps, lrs)

def onetenth_20_30(lr):
    steps = [20, 30]
    lrs = [lr, lr / 10, lr / 100]
    return Step(steps, lrs)


def wideresnet_step(lr):
    steps = [60, 120, 160]
    lrs = [lr, lr / 5, lr / 25, lr / 125]
    return Step(steps, lrs)
