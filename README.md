# HIDSI Workshop — Differentiable Optimization with JAX on Koa

Materials from the HIDSI Workshop Series session on JAX and differentiable optimization
(University of Hawaiʻi, October 1, 2026), presented by William Kwonho Paek.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/HIDSI_JAX_workshop/blob/main/notebooks/Oct1JAXworkshop.ipynb)

The fastest way to follow along is the button above: it opens the demo notebook in Google Colab,
with nothing to install.

## What the workshop covered

| # | Module | Material |
|---|--------|----------|
| 1 | **Live demo**: JAX solves and differentiates a small problem | [`notebooks/Oct1JAXworkshop.ipynb`](notebooks/Oct1JAXworkshop.ipynb) |
| 2 | **What can JAX do?** `grad`, `jit`, `vmap`, and putting a solver inside a model | [`slides/What_Can_JAX_Do.pdf`](slides/What_Can_JAX_Do.pdf) |
| 3 | **Math meets technology**: a short history of differentiable optimization, and where mathematics and AI stand in 2026 | [`slides/History_of_Differentiable_Optimization.pdf`](slides/History_of_Differentiable_Optimization.pdf), [`slides/Mathematics_AI_2026_Turning_Point.pdf`](slides/Mathematics_AI_2026_Turning_Point.pdf) |
| 4 | **Q&A**: discussion of participants' own research problems | — |

## The demo notebook

Run the cells from top to bottom. Each one introduces one idea:

1. **Setup**: import JAX, `jax.numpy`, and matplotlib.
2. **`jax.grad`**: the exact derivative of `f(x) = x²`, and the second derivative by applying `grad` twice.
3. **Why `grad` needs a scalar**: `jax.grad(jnp.sin)` works on one number but raises an error on an array. *This error is intentional.*
4. **`jax.vmap`**: wrapping `grad` in `vmap` fixes it, and plots sin(x) next to its derivative.
5. **`jax.jit`**: compiles a loop-heavy function and times it against the uncompiled version on a million inputs.
6. **Gradient descent**: uses `jit(grad(loss))` to walk downhill on `x⁴ − 3x² + x + 5` and plots the path.

## Running it yourself

### Option A: Google Colab (recommended)
Click the badge at the top. Colab already has JAX installed, and the default CPU runtime is enough.
In the saved run (Colab CPU), the `jit` timing cell showed about 358 ms per call uncompiled versus
about 182 ms compiled; your numbers will vary with the hardware.

### Option B: your own computer
```bash
git clone https://github.com/YOUR-USERNAME/HIDSI_JAX_workshop.git
cd HIDSI_JAX_workshop
pip install -r requirements.txt
jupyter notebook notebooks/Oct1JAXworkshop.ipynb
```

## Going further

The slides introduce *implicit differentiation*: getting the gradient of a solver's answer x*
from the condition it satisfies, rather than by recording every solver step. These libraries
put that into practice:

- [JAX](https://github.com/jax-ml/jax): `grad`, `jit`, `vmap`, `lax.custom_root`
- [Optax](https://github.com/google-deepmind/optax): gradient optimizers (Adam, SGD, …) for training the outer model
- [Optimistix](https://github.com/patrick-kidger/optimistix): root finding, least squares and minimization with implicit gradients built in
- [JAXopt](https://github.com/google/jaxopt): the earlier implicit-diff library (no longer maintained; useful for reading older papers)

Key papers from the slides: Amos & Kolter, *OptNet* (2017); Agrawal et al., *Differentiable Convex
Optimization Layers* (2019); Bai, Kolter & Koltun, *Deep Equilibrium Models* (2019); Blondel et al.,
*Efficient and Modular Implicit Differentiation* (2021).

## Repository layout
```
HIDSI_JAX_workshop/
├── notebooks/
│   └── Oct1JAXworkshop.ipynb    # live demo (saved from Colab)
├── slides/
│   ├── What_Can_JAX_Do.pdf
│   ├── History_of_Differentiable_Optimization.pdf
│   └── Mathematics_AI_2026_Turning_Point.pdf
├── requirements.txt
└── README.md
```

Thanks to the Hawaiʻi Data Science Institute (HIDSI) for hosting the workshop series.
