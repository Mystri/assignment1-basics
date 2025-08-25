from math import sqrt
from typing import Callable, Optional
import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    ):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            # Group of parameters. Used when we want to apply different hyperparams to
            # Different layers.
            # In our basic network, there are most likely 1 group.
            for p in group["params"]:

                g = p.grad
                state = self.state[p]

                t = self.state.get("t", 1)
                v = self.state.get("v", torch.zeros_like(g))
                m = self.state.get("m", torch.zeros_like(g))

                state["m"] = b1 * m + b1 * (1 - b1) * g
                state["v"] = b2 * v + (1 - b2) * g.pow(2)

                alpha_t = lr * sqrt(1 - pow(b2, t)) / 1 - pow(b1, t)

                # Going at the negative direction of the gradient.
                torch.addcdiv(
                    p.data, value=-alpha_t, tensor1=m, tensor2=torch.sqrt(v) + eps
                )

                state["t"] = t + 1


if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    a = AdamW([weights])
    a.step()
