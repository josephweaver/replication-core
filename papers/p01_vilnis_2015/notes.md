# Vilnis et al. (2015) — Gaussian Word Embeddings

## Core Idea

Replace point-vector word embeddings with Gaussian distributions.

Instead of:

$$
w \in \mathbb{R}^d
$$

represent each word as:

$$
P_w(x)
=
\mathcal N(x;\mu_w,\Sigma_w)
$$

where:

- $$\mu_w \in \mathbb R^d$$ is the semantic location
- $$\Sigma_w \succ 0$$ represents semantic uncertainty/spread

Typically:

$$
\Sigma_w
=
\operatorname{diag}(\sigma_{w,1}^2,\dots,\sigma_{w,d}^2)
$$

with trainable log-variances:

$$
\sigma_{w,i}^2 = \exp(s_{w,i})
$$

to guarantee positivity.

---

# Training Setup

Given corpus sentence:

$$
S=(s_1,\dots,s_n)
$$

For center word:

$$
y=s_i
$$

collect nearby context words:

$$
x \in \mathcal C(y)
$$

using a Skip-Gram sliding window.

Sample negative words:

$$
z \sim q(z)
$$

from a noise distribution.

Training examples become triples:

$$
(y,x,z)
$$

where:

- $$(y,x)$$ is positive
- $$(y,z)$$ is negative

---

# Gaussian Similarity

Instead of dot products:

$$
v_y^\top v_x
$$

compare Gaussian overlap:

$$
E(y,x)
=
\int
\mathcal N(t;\mu_y,\Sigma_y)
\mathcal N(t;\mu_x,\Sigma_x)
\,dt
$$

This has closed form:

$$
E(y,x)
=
\mathcal N(
0;
\mu_y-\mu_x,
\Sigma_y+\Sigma_x
)
$$

Usually optimize log-energy:

$$
\log E(y,x)
=
-\frac12
(\mu_y-\mu_x)^T
(\Sigma_y+\Sigma_x)^{-1}
(\mu_y-\mu_x)
-\frac12
\log\det(\Sigma_y+\Sigma_x)
+ C
$$

---

# Loss Function

Use ranking/max-margin loss:

$$
L
=
\max(
0,
m
-
\log E(y,x)
+
\log E(y,z)
)
$$

Goal:

positive context similarity should exceed negative similarity by margin $$m$$.

---

# Optimization

Train parameters:

$$
\theta_w
=
(\mu_w,s_w)
$$

using SGD/AdaGrad.

Update jointly:

$$
\mu_w
\leftarrow
\mu_w
-
\eta
\frac{\partial L}{\partial \mu_w}
$$

$$
s_w
\leftarrow
s_w
-
\eta
\frac{\partial L}{\partial s_w}
$$

Covariances are constrained to remain bounded:

$$
mI \prec \Sigma_w \prec MI
$$

to avoid:
- covariance collapse
- exploding variances
- unstable inverses/determinants

---

# Interpretation

Means encode semantic location:

$$
\mu_w
$$

Covariances encode:
- semantic breadth
- uncertainty
- ambiguity
- polysemy

Examples:
- broad words → larger covariance
- precise words → smaller covariance

---

# Important Limitation

A single Gaussian is unimodal.

If a word has multiple meanings:

$$
\text{rock}
=
\{\text{music},\text{stone}\}
$$

then optimization may produce:

- mean near midpoint between semantic modes
- inflated covariance covering both meanings

Thus covariance may represent:
- true uncertainty
- AND unresolved multimodal structure

rather than pure variance alone.

This motivated later work using Gaussian mixture embeddings:

$$
p(x\mid w)
=
\sum_{k=1}^K
\pi_k
\mathcal N(\mu_k,\Sigma_k)
$$

---

# Key Conceptual Point

Vilnis is NOT Bayesian inference.

The Gaussian is:
- a learned semantic representation
- not a posterior distribution

The model directly optimizes Gaussian parameters for predictive/discriminative performance.