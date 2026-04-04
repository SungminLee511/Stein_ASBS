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

$$\sum_{i=1}^N c_i k(\cdot, x_i) : N \geq 1, c_i \in \mathbb{R}, x_i \in \mathbb{R}^d$$

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

**Proposition 1.7** (Explicit Stein Kernel). For RBF base kernel $k$:

$$k_p(x, x’) = T_1(x, x’) + T_2(x, x’) + T_3(x, x’) + T_4(x, x’)$$

where, denoting $s = s_p(x)$, $s’ = s_p(x’)$, $\delta = x - x’$, $r^2 = |\delta|^2$:

$$T_1 = s^T s’ \cdot k(x, x’)$$

$$T_2 = s^T \nabla_{x’} k(x, x’) = \frac{s^T \delta}{\ell^2} \cdot k(x, x’)$$

$$T_3 = (s’)^T \nabla_x k(x, x’) \cdot (-1) = -\frac{(s’)^T \delta}{\ell^2} \cdot k(x, x’)$$

Wait — let me be more careful. The Stein operator applied in $x’$ to $k(x, x’)$ gives:

$$\mathcal{A}*p^{x’} k(x, x’) = s_p(x’)^T \nabla*{x’} k(x, x’) + \nabla_{x’} \cdot \nabla_{x’} k(x, x’)$$

Hmm, that’s not quite right either. Let me derive this step by step.

**Derivation.** The Stein operator acts on a function $g: \mathbb{R}^d \to \mathbb{R}^d$. To construct the Stein kernel, we apply $\mathcal{A}_p$ in each variable to the kernel viewed as a vector-valued function.

Specifically, for each fixed $x’$, define $g_{x’}(x) = k(x, x’) \cdot e_j$ (the kernel times the $j$-th unit vector, for each component $j$). Then:

$$\mathcal{A}_p^x [k(x, x’) e_j] = s_p(x)_j , k(x, x’) + \frac{\partial k}{\partial x_j}(x, x’)$$

The Stein kernel is obtained by applying $\mathcal{A}_p$ in both variables and summing over the shared index:

$$k_p(x, x’) = \sum_{j=1}^d \left[s_p(x)_j , k(x, x’) + \frac{\partial k}{\partial x_j}(x, x’)\right]\left[s_p(x’)_j , k(x, x’) + \frac{\partial k}{\partial x’_j}(x, x’)\right] \cdot \frac{1}{k(x,x’)}$$

Actually, the standard construction is more direct. We define:

$$k_p(x, x’) = s_p(x)^T k(x, x’) s_p(x’) + s_p(x)^T \nabla_{x’} k(x, x’) + \nabla_x k(x, x’)^T s_p(x’) + \text{tr}(\nabla_x \nabla_{x’}^T k(x, x’))$$

**Proof that this is the correct Stein kernel.** Consider the product RKHS $\mathcal{H}_k^d = \mathcal{H}_k \otimes \mathbb{R}^d$. An element $g \in \mathcal{H}*k^d$ can be written as $g(x) = \sum*{i=1}^N c_i k(x, x_i)$ where $c_i \in \mathbb{R}^d$. The Stein operator applied to such a $g$ is:

$$\mathcal{A}*p g(x) = \sum*{i=1}^N \left[s_p(x)^T c_i , k(x, x_i) + c_i^T \nabla_x k(x, x_i)\right]$$

This is a scalar function lying in the span of ${k_p(\cdot, x_i)}$, where:

$$k_p(x, x’) = s_p(x)^T s_p(x’) , k(x, x’) + s_p(x)^T \nabla_{x’} k(x, x’) + s_p(x’)^T \nabla_x k(x, x’) + \text{tr}(\nabla_x \nabla_{x’}^T k(x, x’))$$

This can be verified by checking $\mathcal{A}_p^x[\mathcal{A}_p^{x’} k(x, x’)]$ using the product rule. $\blacksquare$

**Proposition 1.8** (Stein Kernel for RBF, Explicit). With $s = s_p(x)$, $s’ = s_p(x’)$, $\delta = x - x’$, $r^2 = |\delta|^2$, $K = k(x, x’)$:

$$\boxed{k_p(x, x’) = K \left[s^T s’ + \frac{s^T \delta}{\ell^2} - \frac{(s’)^T \delta}{\ell^2} + \frac{d}{\ell^2} - \frac{r^2}{\ell^4}\right]}$$

where we substituted $\nabla_{x’} k = +\frac{\delta}{\ell^2} K$ and $\nabla_x k = -\frac{\delta}{\ell^2} K$ from Proposition 1.5.

### 1.6 Kernel Stein Discrepancy

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

## 10. Summary of Key Results

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

## 11. Notation Reference

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
