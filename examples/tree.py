import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

@jdc.pytree_dataclass
class Point2D:
    x: float
    y: float

p = Point2D(1.0, 2.0)

def norm(p: Point2D) -> float:
    return jnp.sqrt(p.x ** 2 + p.y ** 2)

# 自動微分 OK！
grad_fn = jax.grad(norm)
print(grad_fn(p))  # → Point2D(x=1.0/√5, y=2.0/√5)
