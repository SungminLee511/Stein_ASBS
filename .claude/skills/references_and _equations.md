# Stein Variational Adjoint Samplers: KSD-Augmented Stochastic Optimal Control

## Full Mathematical Specification with Proofs

-----

## 1. Preliminaries

### 1.1 Setting

Let $p(x) = \frac{1}{Z}\exp(-E(x))$ be a target Boltzmann distribution on $\mathcal{X} = \mathbb{R}^d$ with energy function $E: \mathbb{R}^d \to \mathbb{R}$. We assume:

- (A1) $E$ is twice continuously differentiable: $E \in C^2(\mathbb{R}^d)$
- (A2) $p(x) > 0$ for all $x \in \mathbb{R}^d$ (full support)
- (A3) $p(x) \to 0$ sufficiently fast as $|x| \to \infty$ (boundary decay)
- (A4) The normalizing constant $Z = \int \exp(-E(x)),dx < \infty$

The **score function** of $p$ is:

$$s_p(x) = \nabla_x \log p(x) = -\nabla_x E(x) \in \mathbb{R}^d$$

Note: $s_p(x)$ is computable from $E(x)$ via autodiff without knowing $Z$.

### 1.2 The Stein Operator

**Definition 1.1** (Langevin Stein Operator). For a smooth vector-valued function $g: \mathbb{R}^d \to \mathbb{R}^d$, the Stein operator associated with $p$ is:

$$\mathcal{A}*p g(x) = \sum*{j=1}^d s_p(x)*j , g_j(x) + \sum*{j=1}^d \frac{\partial g_j}{\partial x_j}(x) = s_p(x)^T g(x) + \nabla_x \cdot g(x)$$

This is a scalar-valued function: $\mathcal{A}_p g: \mathbb{R}^d \to \mathbb{R}$.

**Theorem 1.2** (Stein’s Identity). Under assumptions (A1)–(A3), for any $g \in C^1(\mathbb{R}^d; \mathbb{R}^d)$ satisfying $\lim_{|x| \to \infty} p(x) g(x) = 0$:

$$\mathbb{E}_p[\mathcal{A}_p g(X)] = 0$$

**Proof.** Expand the expectation:

$$\mathbb{E}_p[\mathcal{A}*p g(X)] = \int*{\mathbb{R}^d} p(x) \left[s_p(x)^T g(x) + \nabla \cdot g(x)\right] dx$$

Substitute $s_p(x) = \nabla \log p(x) = \frac{\nabla p(x)}{p(x)}$:

$$= \int_{\mathbb{R}^d} p(x) \frac{\nabla p(x)^T}{p(x)} g(x),dx + \int_{\mathbb{R}^d} p(x) \nabla \cdot g(x),dx$$

$$= \int_{\mathbb{R}^d} \nabla p(x)^T g(x),dx + \int_{\mathbb{R}^d} p(x) \nabla \cdot g(x),dx$$

$$= \int_{\mathbb{R}^d} \left[\nabla p(x)^T g(x) + p(x) \nabla \cdot g(x)\right] dx$$

By the product rule for divergence: $\nabla \cdot (p g) = \nabla p^T g + p \nabla \cdot g$. Therefore:

$$= \int_{\mathbb{R}^d} \nabla \cdot (p(x) g(x)),dx$$

By the divergence theorem:

$$= \lim_{R \to \infty} \oint_{|x| = R} p(x) g(x) \cdot \hat{n},dS = 0$$

The last equality follows from (A3): $p(x) g(x) \to 0$ as $|x| \to \infty$. $\blacksquare$

### 1.3 Reproducing Kernel Hilbert Spaces

**Definition 1.3** (Positive Definite Kernel). A symmetric function $k: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$ is positive definite if for all $N \geq 1$, all $x_1, \ldots, x_N \in \mathbb{R}^d$, and all $c_1, \ldots, c_N \in \mathbb{R}$:

$$\sum_{i=1}^N \sum_{j=1}^N c_i c_j k(x_i, x_j) \geq 0$$

**Definition 1.4** (RKHS). The reproducing kernel Hilbert space $\mathcal{H}_k$ associated with a positive definite kernel $k$ is the closure of:

$$\left{\sum_{i=1}^N c_i k(\cdot, x_i) : N \geq 1, c_i \in \mathbb{R}, x_i \in \mathbb{R}^d\right}$$

under the inner product $\langle k(\cdot, x), k(\cdot, y)\rangle_{\mathcal{H}_k} = k(x, y)$.

**Reproducing property:** For all $f \in \mathcal{H}*k$: $f(x) = \langle f, k(\cdot, x)\rangle*{\mathcal{H}_k}$.

### 1.4 The RBF Kernel and Its Derivatives

Throughout this document we use the **radial basis function (RBF) kernel**:

$$k(x, x’) = \exp\left(-\frac{|x - x’|^2}{2\ell^2}\right)$$

with bandwidth $\ell > 0$.

**Proposition 1.5** (RBF kernel derivatives). Let $\delta = x - x’ \in \mathbb{R}^d$ and $r^2 = |\delta|^2$. Then:

**(a)** First derivative with respect to $x$:

$$\frac{\partial k}{\partial x_i}(x, x’) = -\frac{\delta_i}{\ell^2},k(x, x’)$$

In vector form: $\nabla_x k(x, x’) = -\frac{\delta}{\ell^2},k(x, x’)$

**(b)** First derivative with respect to $x’$:

$$\frac{\partial k}{\partial x’_i}(x, x’) = \frac{\delta_i}{\ell^2},k(x, x’)$$

In vector form: $\nabla_{x’} k(x, x’) = \frac{\delta}{\ell^2},k(x, x’)$

**(c)** Mixed second derivatives:

$$\frac{\partial^2 k}{\partial x_i \partial x’*j}(x, x’) = \frac{1}{\ell^2}\left(\delta*{ij} - \frac{\delta_i \delta_j}{\ell^2}\right) k(x, x’)$$

where $\delta_{ij}$ is the Kronecker delta.

**(d)** Trace of mixed second derivatives:

$$\text{tr}!\left(\nabla_x \nabla_{x’}^T k(x, x’)\right) = \sum_{i=1}^d \frac{\partial^2 k}{\partial x_i \partial x’_i} = \frac{1}{\ell^2}\left(d - \frac{r^2}{\ell^2}\right) k(x, x’)$$

**(e)** Second derivative with respect to $x$ only:

$$\frac{\partial^2 k}{\partial x_i \partial x_j}(x, x’) = \frac{1}{\ell^2}\left(\frac{\delta_i \delta_j}{\ell^2} - \delta_{ij}\right)k(x, x’)$$

**(f)** Gradient of the first derivative (needed for Stein kernel gradient):

$$\frac{\partial}{\partial x_m}\left[\frac{\partial k}{\partial x_i}\right] = \frac{1}{\ell^2}\left(\frac{\delta_m \delta_i}{\ell^2} - \delta_{mi}\right) k(x, x’)$$

**Proof of (a).** Direct computation:

$$\frac{\partial k}{\partial x_i} = \frac{\partial}{\partial x_i}\exp\left(-\frac{\sum_j (x_j - x’_j)^2}{2\ell^2}\right) = \exp(\cdots) \cdot \left(-\frac{2(x_i - x’_i)}{2\ell^2}\right) = -\frac{\delta_i}{\ell^2},k$$

Parts (b)–(f) follow by analogous direct computation. $\blacksquare$

### 1.5 The Stein Kernel

**Definition 1.6** (Stein Kernel). Given a positive definite kernel $k$ and a target distribution $p$ with score $s_p$, the Stein kernel is:

$$k_p(x, x’) = \mathcal{A}_p^x \mathcal{A}_p^{x’} k(x, x’)$$

where $\mathcal{A}_p^x$ applies the Stein operator in the $x$ variable (treating $x’$ as fixed), and $\mathcal{A}_p^{x’}$ applies it in the $x’$ variable.

**Proposition 1.7** (Stein Kernel, General Form). The Stein kernel is defined as:

$$k_p(x, x’) = s_p(x)^T k(x, x’) s_p(x’) + s_p(x)^T \nabla_{x’} k(x, x’) + \nabla_x k(x, x’)^T s_p(x’) + \text{tr}(\nabla_x \nabla_{x’}^T k(x, x’))$$

**Proof.** Consider the product RKHS $\mathcal{H}_k^d = \mathcal{H}_k \otimes \mathbb{R}^d$. An element $g \in \mathcal{H}*k^d$ can be written as $g(x) = \sum*{i=1}^N c_i k(x, x_i)$ where $c_i \in \mathbb{R}^d$. The Stein operator applied to such a $g$ is:

$$\mathcal{A}*p g(x) = \sum*{i=1}^N \left[s_p(x)^T c_i , k(x, x_i) + c_i^T \nabla_x k(x, x_i)\right]$$

This is a scalar function in the RKHS induced by the kernel $k_p(x, x’)$ defined above: the reproducing property $\mathcal{A}_p g(x) = \langle \mathcal{A}_p g, k_p(\cdot, x)\rangle$ can be verified by expanding $\mathcal{A}_p^x[\mathcal{A}_p^{x’} k(x, x’)]$ via the product rule and collecting terms. $\blacksquare$

**Proposition 1.8** (Stein Kernel for RBF, Explicit). Substituting the RBF derivatives from Proposition 1.5 into Proposition 1.7, with $s = s_p(x)$, $s’ = s_p(x’)$, $\delta = x - x’$, $r^2 = |\delta|^2$, $K = k(x, x’)$:

$$\boxed{k_p(x, x’) = K \left[s^T s’ + \frac{s^T \delta}{\ell^2} - \frac{(s’)^T \delta}{\ell^2} + \frac{d}{\ell^2} - \frac{r^2}{\ell^4}\right]}$$

**Proof.** Substitute each derivative from Proposition 1.5:

- $s^T k, s’ = s^T s’ \cdot K$ (first term)
- $s^T \nabla_{x’} k = s^T \frac{\delta}{\ell^2} K = \frac{s^T \delta}{\ell^2} K$ (second term, using Prop 1.5(b))
- $\nabla_x k^T s’ = (-\frac{\delta}{\ell^2} K)^T s’ = -\frac{(s’)^T \delta}{\ell^2} K$ (third term, using Prop 1.5(a))
- $\text{tr}(\nabla_x \nabla_{x’}^T k) = \frac{1}{\ell^2}(d - \frac{r^2}{\ell^2}) K$ (fourth term, using Prop 1.5(d))

Factor out $K$. $\blacksquare$

### 1.6 Kernel Stein Discrepancy

**Intuition.** The Kernel Stein Discrepancy (KSD) answers the question: “Given a bag of samples, how well do they represent the target distribution $p$ — using only the score $\nabla \log p$, without knowing the normalizing constant $Z$ or having reference samples from $p$?”

The idea is simple. By Stein’s identity, $\mathbb{E}_p[\mathcal{A}_p g(X)] = 0$ for any test function $g$. If we instead compute $\mathbb{E}_q[\mathcal{A}_p g(X)]$ using samples from some other distribution $q$, the result is generally nonzero. The KSD measures the *worst-case* violation: how large can $\mathbb{E}_q[\mathcal{A}_p g]$ be, over all test functions $g$ in the unit ball of the RKHS? If this worst-case violation is zero, then $q = p$. If it’s large, then $q$ is far from $p$.

Crucially, computing KSD requires only the score $s_p(x) = -\nabla E(x)$ evaluated at the sample locations — no density evaluations, no reference samples, no normalizing constant. This makes it the ideal diagnostic for SDE-based samplers like ASBS, where the model density $q_\theta(x)$ is intractable.

**Definition 1.9** (KSD). The kernel Stein discrepancy between a distribution $q$ and the target $p$ is:

$$\text{KSD}^2(q, p) = \mathbb{E}_{X, X’ \overset{\text{iid}}{\sim} q}\left[k_p(X, X’)\right]$$

**Proposition 1.10** (KSD characterizes $p$). Under mild conditions on the kernel $k$ (e.g., $k$ is $C_0$-universal), $\text{KSD}^2(q, p) = 0$ if and only if $q = p$.

**Proof sketch.** $\text{KSD}^2(q, p) = |\mu_q^{\text{Stein}}|*{\mathcal{H}*{k_p}}^2$ where $\mu_q^{\text{Stein}} = \mathbb{E}_q[\mathcal{A}_p k_p(\cdot, X)]$ is the Stein kernel mean embedding. This vanishes iff $\mathbb{E}_q[\mathcal{A}_p g] = 0$ for all $g \in \mathcal{H}_k^d$, which by Stein’s characterization implies $q = p$. See Chwialkowski et al. (2016), Liu et al. (2016) for details. $\blacksquare$

**Proposition 1.11** (U-statistic estimator). Given i.i.d. samples $X_1, \ldots, X_N \sim q$:

$$\widehat{\text{KSD}}^2 = \frac{1}{N(N-1)} \sum_{i \neq j} k_p(X_i, X_j)$$

is an unbiased estimator of $\text{KSD}^2(q, p)$.

-----

## 2. The ASBS Stochastic Optimal Control Framework

### 2.1 The Controlled SDE

ASBS defines a controlled stochastic process on $[0, 1]$:

$$dX_t = \left[f(X_t, t) + g(t)^2 u_\theta(X_t, t)\right]dt + g(t),dW_t, \qquad X_0 \sim \mu$$

where:

- $f: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$ is the reference drift (e.g., $f = 0$ for VE-SDE)
- $g: [0,1] \to \mathbb{R}$ is the diffusion coefficient
- $u_\theta: \mathbb{R}^d \times [0,1] \to \mathbb{R}^d$ is the learned controller (neural network)
- $\mu$ is the source distribution
- $W_t$ is a standard $d$-dimensional Wiener process

We write the total drift as $b_\theta(x, t) = f(x, t) + g(t)^2 u_\theta(x, t)$.

### 2.2 The Standard SOC Problem

The standard ASBS solves:

$$\min_{u_\theta} ; J_0[u_\theta] = \mathbb{E}\left[\frac{1}{2}\int_0^1 |u_\theta(X_t, t)|^2,dt + \Phi_0(X_1)\right]$$

where $\Phi_0(x) = E(x)$ (for AS) or $\Phi_0(x) = E(x) - \log h_\psi(x, 1)$ (for ASBS with corrector $h_\psi$).

### 2.3 Pontryagin’s Maximum Principle

**Theorem 2.1** (Adjoint Equations for Standard ASBS). The optimal control satisfies $u^*(x, t) = g(t) Y_t$ where the adjoint process $Y_t \in \mathbb{R}^d$ obeys:

**Terminal condition:**
$$Y_1 = -\nabla_{X_1} \Phi_0(X_1) = -\nabla E(X_1) \quad (\text{for AS})$$

**Backward dynamics:**
$$dY_t = -\left[(\nabla_x f(X_t, t))^T Y_t\right]dt + d\tilde{W}_t$$

where $\tilde{W}_t$ is a backward martingale term (irrelevant for the matching objective).

**For VE-SDE** ($f = 0$, driftless reference): The backward dynamics reduce to $dY_t = 0 + d\tilde{W}_t$, so $\mathbb{E}[Y_t \mid X_0, X_1] = Y_1$ is constant in time. The adjoint is simply the terminal energy gradient.

### 2.4 The Adjoint Matching Objective

The controller $u_\theta$ is trained via regression:

$$\mathcal{L}*{\text{AM}}(\theta) = \mathbb{E}*{t \sim \text{Uniform}(0,1)} , \mathbb{E}*{X_t \sim p_t^{u*\theta}}\left[|u_\theta(X_t, t) - g(t)(-Y_t)|^2\right]$$

In practice, this is estimated from buffered trajectory data: simulate forward SDE, compute $Y_1 = -\nabla \Phi_0(X_1)$, propagate adjoint backward, store $(t, X_t, Y_t)$ tuples, train $u_\theta$ by regression.

-----

## 3. The KSD-Augmented SOC Problem

### 3.1 Motivation

The standard SOC objective $J_0$ optimizes each particle independently. Given $N$ terminal samples $X_1^1, \ldots, X_1^N$, each minimizes its own $\Phi_0(X_1^i)$. If $\Phi_0$ has a dominant mode, all particles converge there — **mode collapse**.

We augment the terminal cost with a distributional penalty that measures how well the empirical terminal distribution matches $p$.

### 3.2 The Augmented Terminal Cost

**Definition 3.1** (KSD-Augmented Terminal Cost). For $N$ particles with terminal positions $\mathbf{X}_1 = (X_1^1, \ldots, X_1^N) \in \mathbb{R}^{N \times d}$:

$$\Phi(\mathbf{X}*1) = \frac{1}{N}\sum*{i=1}^N \Phi_0(X_1^i) + \frac{\lambda}{2} \cdot \frac{1}{N^2}\sum_{i=1}^N \sum_{j=1}^N k_p(X_1^i, X_1^j)$$

where:

- The first term is the standard per-particle terminal cost (averaged)
- The second term is $\frac{\lambda}{2}\widehat{\text{KSD}}_V^2$ where $\widehat{\text{KSD}}*V^2 = \frac{1}{N^2}\sum*{i,j} k_p(X_1^i, X_1^j)$ is the V-statistic estimator of $\text{KSD}^2$
- $\lambda \geq 0$ is a hyperparameter controlling the strength of the distributional penalty

**Remark.** We use the V-statistic (including diagonal terms) rather than the U-statistic because: (a) the gradient is simpler, and (b) the diagonal terms $k_p(x, x) = |s_p(x)|^2 + \frac{d}{\ell^2}$ are bounded constants that don’t affect the optimization.

### 3.3 The Augmented SOC Problem

$$\min_{u_\theta} ; J[u_\theta] = \mathbb{E}\left[\frac{1}{2}\int_0^1 \frac{1}{N}\sum_{i=1}^N|u_\theta(X_t^i, t)|^2,dt + \Phi(\mathbf{X}_1)\right]$$

subject to each particle following the same controlled SDE independently:

$$dX_t^i = \left[f(X_t^i, t) + g(t)^2 u_\theta(X_t^i, t)\right]dt + g(t),dW_t^i, \qquad X_0^i \overset{\text{iid}}{\sim} \mu$$

**Key structural property:** The forward dynamics are **independent** across particles. The coupling enters **only through the terminal cost** $\Phi(\mathbf{X}_1)$.

-----

## 4. Derivation of the Coupled Adjoint Equations

### 4.1 The Adjoint Terminal Condition

**Theorem 4.1** (Coupled Adjoint Terminal Condition). The adjoint process for particle $i$ has terminal condition:

$$Y_1^i = -\nabla_{X_1^i} \Phi(\mathbf{X}*1) = -\frac{1}{N}\nabla \Phi_0(X_1^i) - \frac{\lambda}{N^2}\sum*{j=1}^N \nabla_x k_p(X_1^i, X_1^j)$$

**Proof.** Compute $\nabla_{X_1^i} \Phi(\mathbf{X}_1)$ by differentiating each term in $\Phi$:

**Term 1:** $\nabla_{X_1^i}\left[\frac{1}{N}\sum_{m=1}^N \Phi_0(X_1^m)\right] = \frac{1}{N}\nabla \Phi_0(X_1^i)$

since only the $m = i$ term depends on $X_1^i$.

**Term 2:** $\nabla_{X_1^i}\left[\frac{\lambda}{2N^2}\sum_{m=1}^N \sum_{n=1}^N k_p(X_1^m, X_1^n)\right]$

The double sum has terms depending on $X_1^i$ in two ways: when $m = i$ (first argument) and when $n = i$ (second argument). By symmetry of $k_p$ ($k_p(x, x’) = k_p(x’, x)$), these contribute equally:

$$= \frac{\lambda}{2N^2}\left[\sum_{n=1}^N \nabla_x k_p(X_1^i, X_1^n) + \sum_{m=1}^N \nabla_{x’} k_p(X_1^m, X_1^i)\right]$$

$$= \frac{\lambda}{2N^2}\left[\sum_{j=1}^N \nabla_x k_p(X_1^i, X_1^j) + \sum_{j=1}^N \nabla_{x’} k_p(X_1^j, X_1^i)\right]$$

By symmetry: $\nabla_{x’} k_p(x’, x)\big|*{x’ = X_1^j, x = X_1^i} = \nabla_x k_p(X_1^j, X_1^i)\big|*{\text{w.r.t. first arg}}$…

Actually, let me be more careful. Since $k_p(x, x’) = k_p(x’, x)$ (the Stein kernel with RBF base is symmetric), we have:

$$\nabla_{x’} k_p(x, x’) = \nabla_x k_p(x’, x)$$

So:

$$\nabla_{X_1^i}\left[\frac{\lambda}{2N^2}\sum_{m,n} k_p(X_1^m, X_1^n)\right] = \frac{\lambda}{2N^2} \cdot 2 \sum_{j=1}^N \nabla_x k_p(X_1^i, X_1^j) = \frac{\lambda}{N^2}\sum_{j=1}^N \nabla_x k_p(X_1^i, X_1^j)$$

Combining:

$$\nabla_{X_1^i} \Phi(\mathbf{X}*1) = \frac{1}{N}\nabla \Phi_0(X_1^i) + \frac{\lambda}{N^2}\sum*{j=1}^N \nabla_x k_p(X_1^i, X_1^j)$$

Therefore:

$$Y_1^i = -\nabla_{X_1^i} \Phi(\mathbf{X}*1) = -\frac{1}{N}\nabla \Phi_0(X_1^i) - \frac{\lambda}{N^2}\sum*{j=1}^N \nabla_x k_p(X_1^i, X_1^j) \qquad \blacksquare$$

### 4.2 The Backward Dynamics Are Decoupled

**Theorem 4.2** (Decoupled Backward Dynamics). The adjoint process for particle $i$ satisfies, for $t \in [0, 1)$:

$$dY_t^i = -\left[(\nabla_x f(X_t^i, t))^T Y_t^i\right]dt + d\tilde{W}_t^i$$

This depends only on particle $i$’s own state $(X_t^i, Y_t^i)$, not on any other particle.

**Proof.** The adjoint backward dynamics are derived from the Hamiltonian of the optimal control problem. The Hamiltonian for particle $i$ at time $t < 1$ is:

$$H^i(x, y, u, t) = \frac{1}{2N}|u|^2 + y^T\left[f(x, t) + g(t)^2 u\right]$$

where $x = X_t^i$, $y = Y_t^i$, $u = u_\theta(X_t^i, t)$. This depends only on particle $i$’s variables — no other particles appear, because the forward dynamics are independent.

The adjoint dynamics are $\dot{Y}_t^i = -\nabla_x H^i = -(\nabla_x f)^T Y_t^i$ (plus the stochastic term from the diffusion). No coupling. $\blacksquare$

**Corollary 4.3** (VE-SDE Case). For VE-SDE ($f = 0$), the adjoint is constant in time:

$$Y_t^i = Y_1^i = -\frac{1}{N}\nabla \Phi_0(X_1^i) - \frac{\lambda}{N^2}\sum_{j=1}^N \nabla_x k_p(X_1^i, X_1^j) \qquad \forall, t \in [0, 1]$$

**Proof.** With $f = 0$: $\dot{Y}_t^i = -(\nabla_x 0)^T Y_t^i = 0$. So $Y_t^i$ is constant and equals its terminal value. $\blacksquare$

-----

## 5. The Gradient of the Stein Kernel

This section derives $\nabla_x k_p(x, x’)$ — the gradient of the Stein kernel with respect to its first argument. This is the key quantity needed for the coupled adjoint terminal condition (Theorem 4.1).

### 5.1 Setup

Recall from Proposition 1.8:

$$k_p(x, x’) = K \left[s^T s’ + \frac{s^T \delta}{\ell^2} - \frac{(s’)^T \delta}{\ell^2} + \frac{d}{\ell^2} - \frac{r^2}{\ell^4}\right]$$

where $s = s_p(x)$, $s’ = s_p(x’)$, $\delta = x - x’$, $r^2 = |\delta|^2$, $K = k(x, x’) = \exp(-r^2/(2\ell^2))$.

Define the scalar factor:

$$\Gamma(x, x’) = s^T s’ + \frac{s^T \delta}{\ell^2} - \frac{(s’)^T \delta}{\ell^2} + \frac{d}{\ell^2} - \frac{r^2}{\ell^4}$$

so that $k_p(x, x’) = K \cdot \Gamma$.

### 5.2 Gradient via Product Rule

$$\nabla_x k_p(x, x’) = \nabla_x K \cdot \Gamma + K \cdot \nabla_x \Gamma$$

We compute each part separately.

### 5.3 Gradient of $K$

From Proposition 1.5(a):

$$\frac{\partial K}{\partial x_m} = -\frac{\delta_m}{\ell^2},K$$

So $\nabla_x K = -\frac{\delta}{\ell^2},K$.

### 5.4 Gradient of $\Gamma$

$\Gamma$ has five terms. We differentiate each with respect to $x_m$.

**Term $\Gamma_1 = s^T s’ = \sum_j s_j(x) , s’_j(x’)$:**

$$\frac{\partial \Gamma_1}{\partial x_m} = \sum_j \frac{\partial s_j}{\partial x_m}(x) , s’*j(x’) = \sum_j H*{mj}(x) , s’_j(x’) = [H(x) , s’]_m$$

where $H(x) = \nabla_x s_p(x) = \nabla_x^2 \log p(x) = -\nabla_x^2 E(x)$ is the Hessian of the log-density.

In vector form: $\nabla_x \Gamma_1 = H(x),s’$

**Term $\Gamma_2 = \frac{s^T \delta}{\ell^2} = \frac{1}{\ell^2}\sum_j s_j(x)(x_j - x’_j)$:**

$$\frac{\partial \Gamma_2}{\partial x_m} = \frac{1}{\ell^2}\left[\sum_j \frac{\partial s_j}{\partial x_m}(x_j - x’_j) + s_m(x)\right] = \frac{1}{\ell^2}\left[(H(x),\delta)_m + s_m\right]$$

In vector form: $\nabla_x \Gamma_2 = \frac{1}{\ell^2}\left[H(x),\delta + s\right]$

**Term $\Gamma_3 = -\frac{(s’)^T \delta}{\ell^2} = -\frac{1}{\ell^2}\sum_j s’_j(x’)(x_j - x’_j)$:**

$$\frac{\partial \Gamma_3}{\partial x_m} = -\frac{s’_m}{\ell^2}$$

In vector form: $\nabla_x \Gamma_3 = -\frac{s’}{\ell^2}$

(The score $s’ = s_p(x’)$ does not depend on $x$, and $\frac{\partial \delta_j}{\partial x_m} = \delta_{jm}$.)

**Term $\Gamma_4 = \frac{d}{\ell^2}$:**

$$\nabla_x \Gamma_4 = 0$$

(Constant in $x$.)

**Term $\Gamma_5 = -\frac{r^2}{\ell^4} = -\frac{|\delta|^2}{\ell^4}$:**

$$\frac{\partial \Gamma_5}{\partial x_m} = -\frac{2\delta_m}{\ell^4}$$

In vector form: $\nabla_x \Gamma_5 = -\frac{2\delta}{\ell^4}$

### 5.5 Assembling the Full Gradient

Combining $\nabla_x \Gamma = \nabla_x \Gamma_1 + \cdots + \nabla_x \Gamma_5$:

$$\nabla_x \Gamma = H(x),s’ + \frac{1}{\ell^2}\left[H(x),\delta + s - s’\right] - \frac{2\delta}{\ell^4}$$

$$= H(x)\left[s’ + \frac{\delta}{\ell^2}\right] + \frac{s - s’}{\ell^2} - \frac{2\delta}{\ell^4}$$

Now the full gradient:

$$\nabla_x k_p(x, x’) = -\frac{\delta}{\ell^2},K,\Gamma + K,\nabla_x \Gamma$$

$$\boxed{\nabla_x k_p(x, x’) = K \left[-\frac{\delta}{\ell^2},\Gamma + H(x)\left(s’ + \frac{\delta}{\ell^2}\right) + \frac{s - s’}{\ell^2} - \frac{2\delta}{\ell^4}\right]}$$

where $\Gamma = s^T s’ + \frac{s^T \delta}{\ell^2} - \frac{(s’)^T \delta}{\ell^2} + \frac{d}{\ell^2} - \frac{r^2}{\ell^4}$.

### 5.6 Simplified Form Without the Hessian

The Hessian $H(x) = -\nabla^2 E(x)$ is expensive to compute ($O(d^2)$ per point). For practical implementation, we can avoid computing $H$ explicitly by using the identity:

$$H(x),v = \nabla_x [s_p(x)^T v] \qquad \text{for any fixed vector } v$$

This is a **Hessian-vector product**, computable via one backward pass through the score.

In our case, we need $H(x)\left(s’ + \frac{\delta}{\ell^2}\right)$. Define $v = s_p(x’) + \frac{x - x’}{\ell^2}$. Then:

$$H(x),v = \nabla_x!\left[s_p(x)^T v\right]\bigg|_{v \text{ fixed}}$$

This is one Jacobian-vector product per pair $(x, x’)$.

### 5.7 The Full Adjoint Terminal Condition (Expanded)

Substituting into Theorem 4.1:

$$Y_1^i = -\frac{1}{N}\nabla \Phi_0(X_1^i) - \frac{\lambda}{N^2}\sum_{j=1}^N \nabla_x k_p(X_1^i, X_1^j)$$

$$= -\frac{1}{N}\nabla \Phi_0(X_1^i) - \frac{\lambda}{N^2}\sum_{j=1}^N K_{ij}\left[-\frac{\delta_{ij}}{\ell^2}\Gamma_{ij} + H_i\left(s_j + \frac{\delta_{ij}}{\ell^2}\right) + \frac{s_i - s_j}{\ell^2} - \frac{2\delta_{ij}}{\ell^4}\right]$$

where we use the shorthand $K_{ij} = k(X_1^i, X_1^j)$, $\delta_{ij} = X_1^i - X_1^j$, $s_i = s_p(X_1^i)$, $s_j = s_p(X_1^j)$, $H_i = H(X_1^i)$, and $\Gamma_{ij} = \Gamma(X_1^i, X_1^j)$.

-----

## 6. The Modified Adjoint Matching Algorithm

### 6.1 What Changes From Standard ASBS

The **only** modification is in the computation of the adjoint terminal condition $Y_1^i$. Everything else — the SDE integration, the backward simulation (for VP-SDE), the buffer, the AM regression, the training loop — remains identical.

Standard ASBS terminal condition:
$$Y_1^i = -\nabla \Phi_0(X_1^i)$$

KSD-augmented terminal condition:
$$Y_1^i = -\frac{1}{N}\nabla \Phi_0(X_1^i) - \frac{\lambda}{N^2}\sum_{j=1}^N \nabla_x k_p(X_1^i, X_1^j)$$

### 6.2 Practical Computation (Avoiding the Hessian)

For each buffer refresh with $N$ terminal samples ${X_1^i}_{i=1}^N$:

**Step 1.** Compute scores: $s_i = s_p(X_1^i) = -\nabla E(X_1^i)$ for all $i$. Cost: $N$ backward passes through $E$.

**Step 2.** Compute pairwise quantities:

- $\delta_{ij} = X_1^i - X_1^j \in \mathbb{R}^d$
- $r_{ij}^2 = |\delta_{ij}|^2$
- $K_{ij} = \exp(-r_{ij}^2 / (2\ell^2))$
- $\Gamma_{ij} = s_i^T s_j + \frac{s_i^T \delta_{ij}}{\ell^2} - \frac{s_j^T \delta_{ij}}{\ell^2} + \frac{d}{\ell^2} - \frac{r_{ij}^2}{\ell^4}$

Cost: $O(N^2 d)$.

**Step 3.** Compute the Hessian-free gradient. For each pair $(i, j)$, the Hessian term $H_i(s_j + \delta_{ij}/\ell^2)$ can be computed as:

$$H_i,v_{ij} = \nabla_{X_1^i}!\left[s_p(X_1^i)^T v_{ij}\right], \qquad v_{ij} = s_j + \frac{\delta_{ij}}{\ell^2}$$

This requires one JVP per pair — $O(N^2)$ JVPs total, which is expensive.

**Practical approximation:** Drop the Hessian terms entirely. The gradient without Hessian is:

$$\nabla_x k_p(x, x’)\bigg|_{\text{no Hess}} = K\left[-\frac{\delta}{\ell^2},\Gamma + \frac{s - s’}{\ell^2} - \frac{2\delta}{\ell^4}\right]$$

This is the gradient of $k_p$ treating the scores $s(x), s(x’)$ as constants (i.e., detaching the score from the computational graph when differentiating $k_p$). This is computationally $O(N^2 d)$ with no JVPs. We call this the **detached Stein kernel gradient**.

**Justification:** The Hessian terms $H_i v_{ij}$ represent the second-order effect of how the score changes as $x$ moves. For well-separated particles (where $K_{ij}$ is small anyway), these terms are negligible. For nearby particles (where $K_{ij}$ is large), the dominant forces are the kernel gradient ($-\delta/\ell^2$) and score difference ($(s-s’)/\ell^2$) terms.

### 6.3 Algorithm Summary

```
KSD-Augmented ASBS Training (single epoch)

Input: controller u_θ, ref_sde, source μ, energy E, λ (KSD weight), ℓ (bandwidth)

1. SAMPLE: Draw N particles x₀¹,...,x₀ᴺ ~ μ
2. FORWARD: Simulate each particle independently:
     x₁ⁱ = sdeint(sde, x₀ⁱ, timesteps)  for i = 1,...,N
3. SCORES: Compute sᵢ = -∇E(x₁ⁱ) for all i
4. STANDARD ADJOINT: Compute a₀ⁱ = -∇Φ₀(x₁ⁱ) for all i
5. KSD CORRECTION: Compute the inter-particle Stein kernel gradient:
     Δᵢ = (λ/N²) Σⱼ ∇ₓ kₚ(x₁ⁱ, x₁ⁱ)  for all i  [using detached gradient]
6. AUGMENTED ADJOINT: Y₁ⁱ = (1/N)·a₀ⁱ + Δᵢ
7. BACKWARD: Propagate adjoint backward (for VP-SDE) or use Y_t = Y₁ (for VE-SDE)
8. BUFFER: Store (t, xₜⁱ, Yₜⁱ) tuples
9. TRAIN: Regress u_θ(xₜ, t) against -Yₜ via MSE loss
```

### 6.4 Computational Overhead

|Component              |Standard ASBS                     |KSD-Augmented ASBS|
|-----------------------|----------------------------------|------------------|
|Forward SDE            |$O(N \cdot T \cdot d)$            |Same              |
|Terminal cost gradient |$O(N \cdot d)$                    |Same              |
|KSD correction (Step 5)|—                                 |$O(N^2 \cdot d)$  |
|Backward simulation    |$O(N \cdot T \cdot d)$            |Same              |
|AM regression          |$O(\text{epochs} \cdot B \cdot d)$|Same              |

The overhead is $O(N^2 d)$ per buffer refresh for the pairwise Stein kernel gradient computation. For $N = 1000$ and $d = 12$ (DW4), this is negligible compared to the SDE integration cost. For $N = 1000$ and $d = 165$ (LJ55), it becomes comparable.

**Optimization:** Use the median heuristic $\ell = \text{median}({r_{ij}})$ once per buffer refresh. Subsample the KSD correction to $M < N$ particles if needed.

**Memory-efficient chunking:** The naive computation of $\Delta_i = \sum_j \nabla_x k_p(x_i, x_j)$ materializes an $(N, N, d)$ tensor costing $O(N^2 d)$ memory. For large $N$ or $d$, we can partition the sum over $j$ into chunks of size $C$:

$$\Delta_i = \sum_{b=1}^{\lceil N/C \rceil} \sum_{j \in \text{chunk}_b} \nabla_x k_p(x_i, x_j)$$

Each chunk requires only $(N, C, d)$ memory. Since addition is associative, the result is mathematically identical — the same loss, the same adjoint correction, the same training dynamics. The only cost is more GPU kernel launches: roughly $\lceil N/C \rceil \times$ the wall-clock time of the gradient computation, but with $\lfloor N/C \rfloor \times$ less peak memory. In practice, the KSD computation is a small fraction of total epoch time (most time is SDE integration and AM regression), so the overhead is minor.

-----

## 7. Mean-Field Analysis

### 7.1 The Mean-Field Limit

As $N \to \infty$, the empirical terminal distribution $\hat{\rho}*N = \frac{1}{N}\sum*{i=1}^N \delta_{X_1^i}$ converges to a population distribution $\rho$. The terminal cost becomes a **functional** of $\rho$:

$$\Phi[\rho] = \int \Phi_0(x),\rho(dx) + \frac{\lambda}{2}\iint k_p(x, x’),\rho(dx),\rho(dx’)$$

$$= \mathbb{E}_\rho[\Phi_0(X)] + \frac{\lambda}{2},\text{KSD}^2(\rho, p)$$

### 7.2 The Mean-Field SOC Problem

$$\min_{u} ;; \mathbb{E}\left[\frac{1}{2}\int_0^1 |u(X_t, t)|^2,dt\right] + \Phi[\text{Law}(X_1)]$$

subject to $dX_t = [f(X_t, t) + g(t)^2 u(X_t, t)]dt + g(t)dW_t$, $X_0 \sim \mu$.

This is a **McKean-Vlasov control problem**: the cost depends on the control $u$ both through the individual trajectory and through the distribution of the terminal state.

### 7.3 Optimality Condition

**Proposition 7.1.** At the mean-field optimum, the control satisfies:

$$u^*(x, t) = g(t),y(x, t)$$

where $y(x, t)$ solves the backward PDE:

$$\partial_t y + (\nabla_x f)^T y = 0$$

with terminal condition:

$$y(x, 1) = -\nabla \Phi_0(x) - \lambda \int \nabla_x k_p(x, x’),\rho^*(dx’)$$

where $\rho^*$ is the optimal terminal distribution.

**Interpretation:** In the population limit, the sum $\frac{1}{N}\sum_j \nabla_x k_p(x, X_1^j)$ is replaced by the integral $\int \nabla_x k_p(x, x’)\rho^*(dx’)$ — the KSD functional derivative evaluated at the optimal distribution.

### 7.4 Fixed Point

At the global optimum, $\rho^* = p$ (the target distribution). Then $\text{KSD}^2(\rho^*, p) = 0$, and the KSD correction term vanishes:

$$\int \nabla_x k_p(x, x’),p(dx’) = \nabla_x \mathbb{E}_{X’ \sim p}[k_p(x, X’)] = \nabla_x \cdot 0 = 0$$

since $\mathbb{E}_p[k_p(x, X’)] = \mathbb{E}_p[\mathcal{A}_p^x \mathcal{A}_p^{X’} k(x, X’)] = \mathcal{A}_p^x \mathbb{E}_p[\mathcal{A}_p^{X’} k(x, X’)] = \mathcal{A}_p^x [0] = 0$ by Stein’s identity.

**Therefore:** At convergence, the KSD-augmented SOC problem reduces to the standard SOC problem. The KSD term acts as a **self-annealing regularizer** — it provides a strong signal early in training (when $\rho$ is far from $p$ and KSD is large) and vanishes at convergence (when $\rho = p$).

-----

## 8. Connection to SVGD

### 8.1 SVGD as Steepest Descent

**Theorem 8.1** (Liu and Wang, 2016). The direction that maximally decreases $\text{KL}(q | p)$ within the unit ball of $\mathcal{H}_k^d$ is:

$$\phi^*(x) = \mathbb{E}_{X \sim q}\left[\mathcal{A}*p^X k(X, x)\right] = \mathbb{E}*{X \sim q}\left[k(X, x),s_p(X) + \nabla_X k(X, x)\right]$$

### 8.2 KSD as SVGD Objective

The KSD² is the squared RKHS norm of the Stein kernel mean embedding. Minimizing KSD² over particle positions is equivalent to performing steepest descent in the Stein geometry. The functional gradient of $\text{KSD}^2(\hat{\rho}_N, p)$ with respect to the position of particle $i$ is:

$$\nabla_{X^i} \text{KSD}*V^2 = \frac{2}{N^2}\sum*{j=1}^N \nabla_x k_p(X^i, X^j)$$

This is (up to constants) exactly the correction term $\Delta_i$ in our augmented adjoint (Step 5 of Section 6.3).

### 8.3 SVGD as a Special Case

If we set $\Phi_0 = 0$ (no per-particle energy cost), $f = 0$ (no reference drift), $g = 0$ (no noise), and optimize only the terminal position via the KSD cost, the SOC problem reduces to:

$$\min_{u} ;\frac{1}{2}\int_0^1 |u|^2 dt + \frac{\lambda}{2}\text{KSD}_V^2$$

The optimal control is $u^*(x, t) = g(t) Y_1$ where $Y_1 = -\frac{\lambda}{N^2}\sum_j \nabla_x k_p(x, X_1^j)$. In the limit of infinitesimal time, this is a single gradient step of SVGD.

Our formulation generalizes this: the per-particle cost $\Phi_0$ ensures each particle reaches a low-energy state, and the KSD cost ensures the collection of particles covers the target distribution. The SOC framework propagates both signals through time, producing a controller that balances energy minimization and distributional coverage throughout the trajectory.

-----

## 9. Bandwidth Selection

### 9.1 The Median Heuristic

$$\ell = \text{median}\left({|X_1^i - X_1^j| : 1 \leq i < j \leq N}\right)$$

This is a standard adaptive bandwidth for kernel methods. It scales with the data spread, ensuring that $K_{ij}$ is neither uniformly near 1 (bandwidth too large) nor uniformly near 0 (bandwidth too small).

### 9.2 Adaptive Bandwidth During Training

Early in training, particles may be spread widely ($\ell$ large, interactions weak). As training progresses and particles concentrate near $p$’s modes, $\ell$ decreases and interactions strengthen. The median heuristic automatically adapts to this.

**Recommendation:** Recompute $\ell$ at each buffer refresh using the current terminal samples.

-----

## 10. Thoughts

### 10.1 Why Mode Collapse Happens in Standard ASBS

The standard SOC objective $J_0$ optimizes each particle independently. Given $N$ terminal samples, each one minimizes its own per-particle cost $\Phi_0(X_1^i)$. If $\Phi_0$ has a dominant mode — a deep energy basin that captures gradient descent from a wide initial basin — all particles converge there. The objective is equally satisfied whether $N$ particles cover 5 modes or crowd into 1, as long as each particle individually achieves low energy.

This is not a bug in the optimization — the objective genuinely has no preference for distributional coverage. It optimizes marginal quality (each sample is good individually) without any joint quality signal (the collection of samples represents $p$ well).

### 10.2 How KSD² Prevents Mode Collapse

The KSD² term provides precisely the missing joint signal. By Stein’s identity, $\mathbb{E}_p[k_p(X, X’)] = 0$: when samples are drawn from $p$, the positive and negative contributions of the Stein kernel cancel on average. The positive contributions (from the $|s_p|^2$ and $d/\ell^2$ terms) are exactly balanced by the negative contributions (from the score-difference and kernel-gradient cross terms) when the empirical distribution matches $p$’s density proportions.

When all particles collapse to a single mode A of a multimodal target, this cancellation fails. The particles are clustered together, their scores are nearly identical ($s_p(x_i) \approx s_p(x_j)$ since all $x_i$ are nearby), the kernel values $K_{ij} \approx 1$, and the $d/\ell^2$ term in $\Gamma$ dominates — producing a large positive KSD². The gradient of KSD² with respect to each particle’s position then provides a force that pushes the empirical distribution back toward $p$.

Critically, this force is **informed by $p$’s geometry** through the score $s_p(x) = -\nabla E(x)$. It doesn’t push particles apart uniformly — it pushes them toward regions where $p$ has mass but the current sample set has no coverage. Particles near a populated mode are pushed toward unpopulated modes, with the direction and magnitude determined by the score landscape.

### 10.3 Why the Stein Kernel Specifically (Not Just Any Batch Interaction)

A natural question is whether any per-batch interaction term would work, or whether the Stein kernel structure is essential. Consider the simplest alternative: a pure kernel repulsion without score information:

$$\Phi_{\text{repulsion}}(\mathbf{X}*1) = \frac{1}{N}\sum_i \Phi_0(X_1^i) + \frac{\lambda}{N^2}\sum*{i,j} k(X_1^i, X_1^j)$$

where $k$ is a plain RBF kernel (no score terms). The gradient of this repulsion pushes particles apart — away from each other — but it has **no knowledge of $p$**. It spreads particles toward uniformity across space, including into regions where $p$ has negligible density. Mode collapse is replaced by a different failure: particles scattered into the void between modes, or in the tails where $p$ is exponentially small.

The Stein kernel $k_p$ avoids this by encoding the target distribution’s structure through the score. The key is the characterization property (Proposition 1.10): $\text{KSD}^2(q, p) = 0$ if and only if $q = p$. Minimizing KSD² doesn’t push toward uniformity — it pushes toward $p$ specifically. No other distribution is a minimizer.

Concretely, the Stein kernel gradient contains terms like $(s_p(x_i) - s_p(x_j))/\ell^2$ — the score difference between two particles. This term is small when particles occupy the same region of score space (same local geometry of $p$) and large when they sit in regions with different score structure (different modes, or a mode vs. a saddle point). A plain kernel’s gradient $\nabla_x k(x, x’) = -\delta/\ell^2 \cdot K$ has no such sensitivity — it only sees distance, blind to $p$’s landscape.

So the answer is that a generic batch interaction would not produce the same effect. The interaction term must encode information about $p$, and the Stein kernel is the canonical construction that does so using only the score $s_p(x)$. Other score-informed interactions would also work (e.g., the SVGD kernel function $k(x,x’)s_p(x’) + \nabla_{x’}k(x,x’)$, which is closely related), but purely geometric repulsion terms would not.

### 10.4 The Self-Annealing Property

An elegant consequence of the Stein construction is that the KSD correction is self-annealing: it provides a strong signal when needed and vanishes when not. At convergence, the terminal distribution $\rho^*$ equals $p$, and $\text{KSD}^2(\rho^*, p) = 0$ by the characterization property. The correction term $\sum_j \nabla_x k_p(x_i, x_j)$ becomes zero, and the augmented objective reduces exactly to the standard ASBS objective.

This means the KSD augmentation cannot hurt a converged model — it only acts when the model is not yet converged, steering it toward better distributional coverage during training. A pure repulsion term, by contrast, would persist even at convergence, actively fighting against the optimal solution.

-----

## 11. Summary of Key Results

|Result                            |Equation                                                                                                                     |Significance                               |
|----------------------------------|-----------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|
|Coupled adjoint terminal condition|$Y_1^i = -\frac{1}{N}\nabla\Phi_0(X_1^i) - \frac{\lambda}{N^2}\sum_j \nabla_x k_p(X_1^i, X_1^j)$                             |Only modification to ASBS                  |
|Backward dynamics decoupled       |$dY_t^i = -(\nabla_x f)^T Y_t^i,dt + d\tilde{W}_t^i$                                                                         |No inter-particle coupling in backward pass|
|VE-SDE simplification             |$Y_t^i = Y_1^i$ constant in time                                                                                             |No backward simulation needed              |
|Stein kernel gradient             |$\nabla_x k_p = K[-\frac{\delta}{\ell^2}\Gamma + H(s’+\frac{\delta}{\ell^2}) + \frac{s-s’}{\ell^2} - \frac{2\delta}{\ell^4}]$|Full gradient with Hessian                 |
|Hessian-free approximation        |Drop $H(\cdot)$ terms, $O(N^2 d)$                                                                                            |Practical implementation                   |
|Mean-field fixed point            |KSD term vanishes when $\rho = p$                                                                                            |Self-annealing regularizer                 |
|Overhead                          |$O(N^2 d)$ per buffer refresh                                                                                                |Negligible for DW4/LJ13                    |

-----

## 12. Notation Reference

|Symbol                                      |Meaning                                  |
|--------------------------------------------|-----------------------------------------|
|$p(x) = Z^{-1}\exp(-E(x))$                  |Target Boltzmann distribution            |
|$s_p(x) = -\nabla E(x)$                     |Score of $p$                             |
|$H(x) = \nabla^2 \log p(x) = -\nabla^2 E(x)$|Hessian of log-density                   |
|$k(x, x’) = \exp(-|x-x’|^2/(2\ell^2))$      |RBF kernel                               |
|$k_p(x, x’)$                                |Stein kernel (Definition 1.6)            |
|$K = k(x, x’)$                              |Shorthand for RBF kernel value           |
|$\delta = x - x’$                           |Pairwise difference                      |
|$r^2 = |\delta|^2$                          |Squared distance                         |
|$\Gamma$                                    |Scalar factor: $k_p = K \cdot \Gamma$    |
|$\ell$                                      |RBF bandwidth                            |
|$\Phi_0(x)$                                 |Per-particle terminal cost               |
|$\Phi(\mathbf{X}_1)$                        |Augmented terminal cost (Definition 3.1) |
|$\lambda$                                   |KSD penalty weight                       |
|$u_\theta(x, t)$                            |Learned controller                       |
|$Y_t^i$                                     |Adjoint process for particle $i$         |
|$\Delta_i$                                  |KSD correction to adjoint of particle $i$|
|$\mathcal{A}_p$                             |Stein operator                           |
|$\text{KSD}^2(q, p)$                        |Kernel Stein discrepancy                 |

-----

## 13. Related Work

### 13.1 Stochastic Optimal Control for Sampling

**Adjoint Matching** (Domingo-Enrich et al., ICLR 2025). Casts reward fine-tuning of diffusion models as stochastic optimal control, proving that a memoryless noise schedule must be enforced. Proposes the Adjoint Matching regression objective that avoids backpropagation through the SDE. This is the foundational training algorithm used in both AS and ASBS, and is the objective we augment with the KSD term.

**Adjoint Sampling (AS)** (Havens et al., ICML 2025). The first on-policy SOC sampler that allows many gradient updates per energy evaluation via a replay buffer. Instantiates Adjoint Matching with the memoryless condition (Dirac delta source), achieving strong results on conformer generation. Our starting point — we address its mode collapse limitation.

**Adjoint Schrödinger Bridge Sampler (ASBS)** (Liu et al., NeurIPS 2025 Oral). Generalizes AS by removing the memoryless condition, enabling arbitrary source distributions via a learned corrector trained by Iterative Proportional Fitting. Our method builds directly on ASBS — we modify only the adjoint terminal condition while inheriting the entire IPF training framework.

**Well-Tempered ASBS (WT-ASBS)** (Meta, 2025). Addresses the same mode collapse problem we address, but via a metadynamics-inspired bias along collective variables (CVs). Requires domain-expert specification of CVs. Our KSD approach is CV-free — it uses only the score $s_p(x) = -\nabla E(x)$, requiring no domain knowledge about which coordinates define the relevant slow modes.

**Stochastic Optimal Control Matching (SOCM)** (Domingo-Enrich et al., NeurIPS 2024). An alternative SOC training algorithm based on matching the Hamilton-Jacobi-Bellman PDE. Related to Adjoint Matching but uses a different loss construction.

**Trust Region SOC** (Blessing et al., 2025). Introduces trust region constraints on the control update to prevent mode collapse in high dimensions. Addresses the same failure mode as our work but through constrained optimization rather than distributional regularization.

**Path Integral Sampler (PIS)** (Zhang and Chen, ICLR 2022). An earlier SOC-based sampler using path integral control theory. Uses log-variance loss rather than Adjoint Matching.

### 13.2 Stein’s Method in Machine Learning

**Stein Variational Gradient Descent (SVGD)** (Liu and Wang, NeurIPS 2016). The foundational work connecting Stein’s method to particle-based sampling. Shows that the steepest descent direction in the RKHS that minimizes $\text{KL}(q | p)$ is $\phi^*(x) = \mathbb{E}_q[k(X, x)s_p(X) + \nabla_X k(X, x)]$. Our KSD terminal cost is directly related — minimizing KSD² over particle positions is equivalent to performing SVGD steps (Section 8 of this document). The key difference is that we embed this within the SOC framework, propagating the distributional signal backward through time via the adjoint.

**Kernel Stein Discrepancy (KSD)** (Chwialkowski et al., ICML 2016; Liu et al., ICML 2016; Gorham and Mackey, NeurIPS 2015). Established KSD as a discrepancy measure for goodness-of-fit testing. Proved the characterization property: KSD = 0 iff $q = p$ under appropriate kernel conditions. We use KSD as the distributional penalty in our augmented terminal cost, relying on this characterization to ensure the penalty drives the terminal distribution toward $p$ specifically.

**Stein Control Variates / Control Functionals** (Oates, Girolami, Chopin, JRSSB 2017). Non-parametric variance reduction for Monte Carlo integration using the Stein operator. Achieves super-root-$N$ convergence rates. The RKHS construction of optimal control variates via $(K_p + \lambda NI)a = f$ directly inspires the Stein CV enhancement in our companion document (MATH_SPEC.md, Enhancement 2).

**Neural Control Variates with Stein** (Belomestny et al., 2024). Provides minimax optimal variance reduction rates for deep neural network-based Stein control variates, overcoming the kernel scaling limitations of RKHS methods. Inspires our Neural Stein CV approach (MATH_SPEC.md, Enhancement 7).

**Scalable Stein Control Variates** (Si, Oates, Duncan, Carin, Briol, 2020). Framework for scaling Stein control variates to large problems via stochastic optimization, unifying polynomial, kernel, and neural approaches. Relevant to our practical implementation decisions.

**Stochastic SVGD** (Nüsken and Richter, 2022). Analyzes the SDE version of SVGD — interacting particles with both deterministic kernel updates and stochastic noise. Develops the cotangent space construction for Stein geometry and establishes correspondence between gradient flow structure and large-deviation principles. Provides theoretical foundation for combining particle interactions with stochastic dynamics, which is what our method does within the SOC framework.

**RBM-SVGD** (Li et al., 2019). Reduces SVGD’s $O(N^2)$ cost via random batch methods for interacting particle systems. Directly relevant to our computational overhead — the same random batch ideas could be applied to our KSD correction for further scaling.

### 13.3 Diffusion Models for Boltzmann Sampling

**Denoising Diffusion Samplers (DDS)** (Vargas et al., 2023). Uses an ergodic Ornstein-Uhlenbeck process as the reference for diffusion-based sampling. One of the first diffusion approaches for Boltzmann sampling.

**Iterated Denoising Energy Matching (iDEM)** (Akhound-Sadegh et al., 2024). Offline, simulation-free training for diffusion samplers using energy matching with a replay buffer. Avoids simulation but introduces bias in score estimates.

**Variance-Tuned Diffusion Importance Sampling (VT-DIS)** (Zhang, Midgley, Hernández-Lobato, TMLR 2025). Post-training method that adapts per-step noise covariance to enable trajectory-wise importance sampling for SDE-based Boltzmann samplers. Achieves unbiased estimation without density evaluation. Addresses the same estimation problem as our Stein CVs (MATH_SPEC.md) but through a complementary mechanism — IS over trajectory space rather than control variates at terminal time.

**Consistency Models for Boltzmann Sampling** (NeurIPS 2024 Workshop). Combines bidirectional consistency models with importance sampling, achieving unbiased samples with 6–25 NFEs. Demonstrates that SDE-to-flow distillation for density evaluation is feasible.

**Progressive Inference-Time Annealing (PITA)** (2025). Combines temperature annealing of the target with diffusion processes for improved mode coverage at inference time. Addresses mode collapse via annealing rather than particle interactions.

**Energy-Based Diffusion Generator (EDG)** (2024). Integrates variational autoencoders with diffusion models for simulation-free Boltzmann sampling. Flexible decoder architecture without bijectivity constraints.

### 13.4 Boltzmann Generators and Normalizing Flows

**Boltzmann Generators** (Noé et al., Science 2019). The foundational framework: train a normalizing flow to approximate $p$, then correct via importance sampling. Requires density evaluation (change-of-variables formula) for IS. Our Stein CV approach (MATH_SPEC.md) can be seen as an alternative correction mechanism that does not require density evaluation.

**Sequential Boltzmann Generators (SBG)** (Tan et al., ICML 2025). Transformer-based invertible flow with continuous-time SMC correction. First Boltzmann sampling of tri- through hexa-peptides in Cartesian coordinates. Represents the current state of the art for exact Boltzmann sampling via flow + correction.

**Prose** (Tan et al., 2025). 285M parameter transferable normalizing flow demonstrating zero-shot transfer to unseen peptide systems. Shows that amortization across molecular systems is possible for flows — the analogous capability for SOC samplers remains open.

**Transferable Boltzmann Generators** (Klein and Noé, NeurIPS 2024). First flow matching model that generalizes across different molecular systems without retraining.

### 13.5 Stein Methods Applied to Diffusion / Score Distillation

**SteinDreamer** (2024). Uses Stein identity-based control variates for variance reduction of score distillation gradients in text-to-3D generation. Same Stein CV concept as our Enhancement 2 (MATH_SPEC.md), but applied to training gradients rather than expectation estimation.

**Stein Diffusion Guidance (SDG)** (2025). Training-free diffusion guidance framework using Stein variational inference to correct approximate posteriors. Uses the Stein operator to steer diffusion samples, but for conditional generation rather than Boltzmann sampling.

**Collaborative Score Distillation (CSD)** (Kim et al., 2023). Uses SVGD-based particle interactions for text-to-3D generation. Conceptually related to our inter-particle KSD correction — both use kernel-based interactions between generated samples to improve distributional quality — but in a completely different domain and without the SOC framework.

### 13.6 Mean-Field Optimal Control

**McKean-Vlasov Control Theory** (Carmona and Delarue, 2018). The mathematical framework for optimal control problems where the cost depends on the distribution of the state, not just the state itself. Our augmented SOC problem (Section 7) is a McKean-Vlasov control problem. The mean-field limit analysis follows the standard theory.

**Propagation of Chaos** (Sznitman, 1991). Classical results on the convergence of interacting particle systems to their mean-field limits. Guarantees that our finite-$N$ implementation approximates the population-level KSD objective as $N \to \infty$.

### 13.7 Summary: What Is Inspired by What

|Our contribution                              |Inspired by                                                           |
|----------------------------------------------|----------------------------------------------------------------------|
|KSD as terminal cost regularizer              |KSD goodness-of-fit testing (Chwialkowski et al., Liu et al., 2016)   |
|Inter-particle gradient in adjoint            |SVGD optimal transport direction (Liu and Wang, 2016)                 |
|Decoupled backward dynamics (Theorem 4.2)     |Pontryagin’s maximum principle applied to independent-particle SOC    |
|Self-annealing property (Section 10.4)        |KSD characterization property ($\text{KSD} = 0 \Leftrightarrow q = p$)|
|Detached Stein kernel gradient (Section 6.2)  |Score detaching in diffusion training (standard practice)             |
|Hessian-free computation via JVP (Section 5.6)|Hutchinson trace estimator and autodiff HVP techniques                |
|Mean-field analysis (Section 7)               |McKean-Vlasov control theory (Carmona and Delarue)                    |
|Addressing mode collapse in ASBS              |WT-ASBS (Meta, 2025), but without requiring collective variables      |
|Memory-efficient chunking (Section 6.4)       |Random batch methods for SVGD (Li et al., 2019)                       |

-----

## 14. WT-ASBS vs Stein: Detailed Comparison

### 14.1 What Are Collective Variables?

A **collective variable (CV)** is a low-dimensional function of the full atomic coordinates that captures the “interesting” slow dynamics of a molecular system. Formally, a CV is a smooth map $\xi: \mathbb{R}^d \to \mathbb{R}^m$ where $d$ is the full dimensionality (e.g., $3 \times N_{\text{atoms}}$) and $m \ll d$ is the CV dimensionality (typically $m = 1, 2$, rarely $m > 3$).

**Example: Alanine dipeptide.** This is a small peptide with 22 atoms ($d = 66$). Its conformational dynamics are almost entirely captured by two backbone dihedral angles:

- $\phi$ (the C–N–Cα–C torsion angle)
- $\psi$ (the N–Cα–C–N torsion angle)

The **Ramachandran plot** — a 2D histogram of $(\phi, \psi)$ — reveals distinct metastable basins: C7eq, C7ax, C5, αR, αL, etc. A molecular system in the C7eq basin can stay there for microseconds before transitioning to C7ax or αR. The dihedral angles $(\phi, \psi)$ are the CVs that distinguish these metastable states.

**Why CVs are powerful when available:** If you know that $(\phi, \psi)$ captures the slow dynamics, you can focus enhanced sampling in this 2D space rather than the full 66D space. Energy barriers between metastable states become barriers in the 2D CV space, which are much easier to overcome. This is the essence of all CV-based enhanced sampling methods.

**Why CVs are a limitation:**

1. **Choosing CVs requires domain expertise.** For alanine dipeptide, generations of computational chemists have established that $(\phi, \psi)$ are the right CVs. For a novel protein or material, the right CVs are unknown.
1. **Wrong CVs lead to wrong results.** If the true slow dynamics involve a coordinate not captured by the chosen CVs, the enhanced sampling method will fail to explore the relevant transitions — it will enhance sampling in the wrong directions.
1. **The number of CVs must be small.** Metadynamics-based methods deposit Gaussian bumps in CV space. In $m$ dimensions, the number of bumps needed to fill the space scales exponentially as $O(\epsilon^{-m})$. For $m > 3$, this becomes computationally prohibitive.
1. **CVs may be non-linear and non-obvious.** For complex systems, the relevant slow coordinates might be non-linear combinations of many atomic positions with no simple physical interpretation. Machine learning methods for CV discovery exist (autoencoders, diffusion maps) but add another layer of complexity and potential failure.

### 14.2 How Classical Metadynamics Works

Metadynamics (Laio and Parrinello, 2002) is an enhanced sampling method from computational chemistry. The core idea:

**During a molecular dynamics simulation, periodically deposit a small repulsive Gaussian bump at the current location in CV space.** Over time, these bumps fill up the energy basins, discouraging the system from revisiting states it has already explored and forcing it over energy barriers into new basins.

Mathematically, the bias potential at time $t$ is:

$$V_{\text{meta}}(\xi, t) = \sum_{t’ < t,; t’ = k\tau} w \cdot \exp\left(-\frac{|\xi - \xi(t’)|^2}{2\sigma^2}\right)$$

where:

- $\xi = \xi(x)$ is the CV value at the current configuration $x$
- $\xi(t’)$ is the CV value at time $t’$ (when a bump was deposited)
- $w$ is the Gaussian height
- $\sigma$ is the Gaussian width
- $\tau$ is the deposition interval

As bumps accumulate, they fill the free energy surface: $V_{\text{meta}}(\xi, t) \to -F(\xi)$ where $F(\xi)$ is the free energy as a function of the CV. The system is eventually pushed out of every basin.

**The problem with standard metadynamics:** The bias never converges — it keeps growing, and the bumps overshoot the free energy. The system oscillates wildly in CV space rather than converging to a flat free energy landscape.

### 14.3 Well-Tempered Metadynamics

**Well-tempered metadynamics** (Barducci, Bussi, Parrinello, PRL 2008) fixes the convergence problem by making the Gaussian height decrease as the bias grows:

$$w(t) = w_0 \cdot \exp\left(-\frac{V_{\text{meta}}(\xi(t), t)}{k_B \Delta T}\right)$$

where $\Delta T$ is a “bias temperature” parameter. As $V_{\text{meta}}$ grows in a region, the bumps deposited there become smaller. Eventually, the bump height goes to zero in well-sampled regions, and the bias converges to:

$$V_{\text{meta}}(\xi, \infty) = -\frac{\Delta T}{T + \Delta T} \cdot F(\xi)$$

This is a **fraction** of the free energy (not the full free energy), which means the system samples from a smoothed distribution rather than a uniform one. The temperature ratio $\Delta T / (T + \Delta T)$ controls how aggressive the flattening is:

- $\Delta T \to 0$: no bias (standard MD)
- $\Delta T \to \infty$: full flattening (standard metadynamics)
- Finite $\Delta T$: well-tempered — smooth convergence

### 14.4 How WT-ASBS Works

WT-ASBS (Meta, 2025) imports the well-tempered metadynamics idea into the ASBS framework. The key modifications:

**Step 1 — Choose CVs.** For alanine dipeptide: $\xi(x) = (\phi(x), \psi(x))$, the backbone dihedral angles. These must be specified by the user.

**Step 2 — Maintain a bias grid.** Discretize the CV space into a grid. Track the accumulated bias $V_{\text{bias}}(\xi)$ on this grid. This is a 2D array (for 2 CVs) that persists across training epochs.

**Step 3 — During each ASBS training epoch:**

- Generate terminal samples ${X_1^i}$ via the standard ASBS SDE
- Compute CV values $\xi_i = \xi(X_1^i)$ for each terminal sample
- Deposit well-tempered Gaussian bumps at each $\xi_i$, updating the bias grid:
  $$V_{\text{bias}}(\xi) \leftarrow V_{\text{bias}}(\xi) + w_0 \exp\left(-\frac{V_{\text{bias}}(\xi_i)}{k_B \Delta T}\right) \exp\left(-\frac{|\xi - \xi_i|^2}{2\sigma^2}\right)$$
- Modify the terminal cost to include the bias:
  $$\Phi_0^{\text{biased}}(x) = E(x) - V_{\text{bias}}(\xi(x))$$

**Step 4 — The adjoint sees the biased energy.** The controller $u_\theta$ is trained with the adjoint terminal condition $Y_1 = -\nabla \Phi_0^{\text{biased}}(X_1)$, which includes the bias gradient. This steers particles away from already-visited CV regions.

**Step 5 — Post-training reweighting.** Because the bias distorts the Boltzmann distribution, the generated samples are not from $p$. To recover correct expectations, samples must be reweighted:
$$w_i \propto \exp\left(\frac{V_{\text{bias}}(\xi(X_1^i))}{k_B T}\right)$$

This reweighting requires knowing $V_{\text{bias}}$, which is stored on the bias grid.

### 14.5 Structural Comparison

|Aspect                            |WT-ASBS                                                                       |KSD-Augmented ASBS (Ours)                                                   |
|----------------------------------|------------------------------------------------------------------------------|----------------------------------------------------------------------------|
|**What it modifies**              |The terminal energy $\Phi_0 \to \Phi_0 - V_{\text{bias}}(\xi(x))$             |The adjoint terminal condition $Y_1^i$                                      |
|**Requires CVs?**                 |**Yes** — must specify $\xi: \mathbb{R}^d \to \mathbb{R}^m$                   |**No** — uses only $s_p(x) = -\nabla E(x)$                                  |
|**History-dependent?**            |**Yes** — bias grid accumulates over training epochs                          |**No** — KSD computed fresh each buffer refresh                             |
|**Interaction between particles?**|**No** — bias depends on the history of visits, not on other current particles|**Yes** — KSD correction for particle $i$ depends on all other particles $j$|
|**Dimensionality of correction**  |Low-$m$ CV space (typically $m = 1, 2$)                                       |Full $d$-dimensional configuration space                                    |
|**Post-training reweighting**     |**Required** — samples are from biased distribution                           |**Not required** — KSD term vanishes at convergence (self-annealing)        |
|**Convergence guarantee**         |Inherits well-tempered metadynamics convergence theory                        |Self-annealing: KSD = 0 at $\rho = p$ (Section 7.4)                         |
|**Computational overhead**        |$O(N \cdot G)$ per epoch ($G$ = grid size)                                    |$O(N^2 \cdot d)$ per buffer refresh                                         |
|**Theoretical framework**         |Enhanced sampling (chemistry)                                                 |Stein’s method + mean-field control (statistics/ML)                         |

### 14.6 Where WT-ASBS Wins

**When good CVs are known and the system is moderate-dimensional**, WT-ASBS is likely superior:

1. **Alanine dipeptide.** The Ramachandran CVs $(\phi, \psi)$ are perfect — they capture essentially all of the relevant slow dynamics. WT-ASBS operates in this well-understood 2D space with proven convergence. Our KSD method operates in the full 66D space with an RBF kernel that may struggle to distinguish modes separated by subtle torsional differences.
1. **Free energy estimation.** WT-ASBS naturally produces a free energy surface as a byproduct of the accumulated bias: $F(\xi) \approx -(T + \Delta T)/\Delta T \cdot V_{\text{bias}}(\xi, \infty)$. Our method produces samples but does not directly estimate free energies (though Stein CVs can help with free energy expectations).
1. **Computational cost.** For low-dimensional CVs ($m = 1, 2$), the bias grid is cheap: $O(G)$ per update where $G$ is the number of grid points (typically $\sim 100 \times 100$). Our KSD correction costs $O(N^2 d)$, which is more expensive for large $N$ and $d$.

### 14.7 Where Our Method Wins (or Could Win)

**1. When CVs are unknown or poorly chosen.**

This is the fundamental advantage. For complex molecular systems — large proteins, disordered materials, multi-component mixtures — the relevant slow coordinates are often unknown. WT-ASBS requires the user to guess or discover CVs before training. If the CVs are wrong, the bias is deposited in an irrelevant subspace, and mode collapse persists in the unbiased degrees of freedom.

Our KSD correction operates in the full configuration space. It doesn’t need to know which coordinates are “slow” — it detects distributional mismatch through the Stein kernel, which incorporates the score $s_p(x) = -\nabla E(x)$ at every particle location. If two modes differ by a subtle rearrangement that no simple CV captures, the score difference $(s_p(x_i) - s_p(x_j))/\ell^2$ in the Stein kernel will still detect it (as long as the kernel bandwidth $\ell$ is appropriate).

**Concrete test cases:**

- **LJ55 (165D, 55 particles).** No one knows good CVs for LJ55. It has $\sim 10^8$ local minima with complex, high-dimensional connectivity. WT-ASBS cannot be meaningfully applied without CVs. Our method can.
- **Rotated synthetic mixtures.** Construct a multimodal energy in $\mathbb{R}^d$ where the modes are separated along a randomly rotated axis. No axis-aligned CV projection separates them. WT-ASBS with any standard CV choice (individual coordinates, principal components) fails. Our KSD method, operating in full space, can detect and correct the mode imbalance.

**2. When the CV space is too high-dimensional.**

If the relevant slow dynamics require $m \geq 3$ CVs, the metadynamics bias grid becomes exponentially expensive ($O(\epsilon^{-m})$ grid points). For $m = 5$ or higher, metadynamics is impractical. Our method has no analogous dimensionality constraint on the “correction space” — it always operates in the full $d$-dimensional space, with cost $O(N^2 d)$ that scales linearly in $d$.

**3. No post-training reweighting needed.**

WT-ASBS generates samples from a biased distribution (the bias $V_{\text{bias}}$ is added to the energy). To recover correct Boltzmann expectations, every observable must be reweighted by $\exp(V_{\text{bias}}(\xi(x)) / k_BT)$. If the bias is large (aggressive exploration), the weights have high variance and the effective sample size drops.

Our method has the self-annealing property (Section 10.4): at convergence, $\text{KSD}^2(\rho, p) = 0$, so the correction term vanishes and the terminal distribution is the unbiased Boltzmann distribution $p$. No reweighting needed. Samples can be used directly for downstream analysis.

**4. A cleaner theoretical framework.**

WT-ASBS imports metadynamics (a heuristic from chemistry with convergence guarantees only in specific settings) into the SOC framework. The connection is engineering: “take a technique that works in MD and apply it to ASBS.” There is no unified theoretical framework explaining why the combination works beyond the individual guarantees of each component.

Our method has a complete theoretical chain: augmented terminal cost (Definition 3.1) → coupled adjoint terminal condition (Theorem 4.1) → decoupled backward dynamics (Theorem 4.2) → mean-field limit (Section 7) → SVGD connection (Section 8) → self-annealing at convergence (Section 7.4). Every step is derived from first principles of optimal control and Stein’s method.

### 14.8 The Experimental Strategy for a Paper

To demonstrate our method’s advantage over WT-ASBS, the experiments should be structured as:

**Experiment 1 — Parity (CVs known).**
Benchmark: alanine dipeptide with Ramachandran CVs.
Goal: Show that KSD-ASBS achieves **comparable** mode coverage to WT-ASBS, despite not using CVs. We likely won’t beat WT-ASBS here — it has the advantage of operating in the correct 2D subspace. But matching its performance without any domain knowledge demonstrates that KSD-ASBS is competitive.

**Experiment 2 — Advantage (CVs unknown).**
Benchmark: LJ55 (no known CVs) or synthetic rotated mixtures.
Goal: Show that KSD-ASBS maintains mode coverage while WT-ASBS with naive CV choices (e.g., first 2 principal components, random projections) fails. This is the key result — demonstrating that our method works where the competitor cannot.

**Experiment 3 — Scaling.**
Benchmark: Synthetic mixtures in $d = 10, 50, 100, 200$.
Goal: Show how mode coverage degrades with dimension for both methods. KSD-ASBS with RBF kernel will eventually degrade (kernel quality decreases in high $d$), but the question is whether it degrades more gracefully than WT-ASBS with poorly chosen CVs.

**Experiment 4 — Ablation.**
Vary $\lambda$ (KSD weight) and show the mode coverage vs. per-particle quality tradeoff. Show that at $\lambda = 0$ (vanilla ASBS), mode collapse occurs; at moderate $\lambda$, mode coverage improves without hurting per-particle quality; at very large $\lambda$, per-particle quality degrades (particles are pushed apart too aggressively).

### 14.9 The Honest Assessment

WT-ASBS is a strong baseline, built by the same team that created ASBS, with access to the most relevant benchmarks and infrastructure. Competing head-to-head on their benchmarks (where CVs are known) is unlikely to produce a clear win for our method.

The path to a publishable result is through **systems where CVs are unavailable** — this is where our method has a structural advantage that no amount of engineering on the WT-ASBS side can overcome. The challenge is making the RBF kernel work well enough in high dimensions to demonstrate this advantage convincingly.

If the RBF kernel degrades too much in high $d$ (which is a real risk for $d > 50$), the natural extension is to replace the RBF kernel with a **learned kernel** — for example, a graph neural network kernel that shares architecture with the EGNN controller. This would be a substantial follow-up contribution but is beyond the scope of the current work.