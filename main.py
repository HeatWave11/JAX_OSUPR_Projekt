import jax
import numpy as np
import jax.numpy as jnp



def f(x):
    y1 = x + x*x + 3
    y2 = x*x + x*x.T
    return y1*y2

x = np.random.randn(3000, 3000).astype('float32')
jax_x_cpu = jax.device_put(jnp.array(x), jax.devices('cpu')[0])

jax_f_cpu = jax.jit(f, backend='cpu')

jax_f_cpu(jax_x_cpu)

timeit -n100 f(x)