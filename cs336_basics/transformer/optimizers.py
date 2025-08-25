import math
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

        # Group of parameters. Used when we want to apply different hyperparams to
        # Different layers.
        # In our basic network, there are most likely 1 group.
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            # There are several layers included in this parameter group. apply optimization for each of them, using that layer 's grad.
            for p in group["params"]:
                g = p.grad

                # Each parameter has a state stored in self.state.
                # Consider self.state[p] as an empty dict, created by pytorch for conveinience.
                state = self.state[p]
                t = state.get("t", 1)
                v = state.get("v", torch.zeros_like(g))
                m = state.get("m", torch.zeros_like(g))

                state["m"] = b1 * m + (1 - b1) * g
                state["v"] = b2 * v + (1 - b2) * g.pow(2)

                alpha_t = lr * (math.sqrt(1 - pow(b2, t)) / (1 - pow(b1, t)))

                # Going at the negative direction of the gradient.
                p.data = torch.addcdiv(
                    p.data, value=-alpha_t, tensor1=state["m"], tensor2=torch.sqrt(state["v"] + eps)
                )

                # Apply weight decay
                if weight_decay != 0:
                    p.data = torch.add(p.data, alpha=-weight_decay * lr, other=p.data, )

                state["t"] = t + 1

        return loss
    
if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    a = AdamW([weights])
    a.step()
