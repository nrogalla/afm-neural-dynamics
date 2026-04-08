from dataclasses import dataclass
from typing import List, Tuple, Optional
import jax.numpy as jnp
import flax.nnx as nnx
from functools import partial
import jax 

@dataclass
class SinusodialPosEmb(nnx.Module):
    dim: int
    def __call__(self, t):
        freqs = jnp.arange(1, self.dim//2 + 1) * jnp.pi
        emb = t * freqs
        return jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)

class TimeConditionedDense(nnx.Module):
    def __init__(self, dim_x: int, dim_h: int, dim_t: int, rngs: nnx.Rngs):
        self.time_emb = SinusodialPosEmb(dim_t)
        self.linear = nnx.Linear(dim_x + dim_h + dim_t, dim_x, rngs=rngs)
    
    def __call__(self, x, h, t):
        t = self.time_emb(t)
        x = jnp.concatenate([x, h, t], axis=-1)
        return self.linear(x)
    
class MLP(nnx.Module): 
    def __init__(self, dim_in, dim_hidden, dim_out, rngs):
        self.py =  nnx.Sequential(nnx.Linear(dim_in, dim_hidden, rngs=rngs),
                                  nnx.gelu,
                                  nnx.Linear(dim_hidden, dim_hidden, rngs=rngs),
                                   nnx.gelu,
                                  nnx.Linear(dim_hidden, dim_out, rngs=rngs))
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.py(x)