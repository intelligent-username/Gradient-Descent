# Gradient Descent

🚧In Progress🚧
TODO:

- Finish the rest of the writeup
- Create the implementations in C++.
- Then, using these implementations, improve the writeup with empirical results.

![Cover](cover.jpg)

---

## Outline

- [Gradient Descent](#gradient-descent)
  - [Outline](#outline)
  - [Motivation](#motivation)
  - [Math](#math)
  - [Variants](#variants)
    - [Batch Gradient Descent (full dataset)](#batch-gradient-descent-full-dataset)
    - [Stochastic Gradient Descent](#stochastic-gradient-descent)
    - [Mini-Batch Gradient Descent](#mini-batch-gradient-descent)
  - [Additional Techniques](#additional-techniques)
    - [Learning Rates](#learning-rates)
      - [1. Fixed](#1-fixed)
      - [2. Scheduled](#2-scheduled)
      - [3. Adaptive](#3-adaptive)
        - [Newton's Method](#newtons-method)
        - [Adagrad](#adagrad)
        - [RMSProp](#rmsprop)
        - [Adam](#adam)
    - [Momentum](#momentum)
      - [Polyak Momentum](#polyak-momentum)
      - [Nesterov Acceleration](#nesterov-acceleration)
    - [Regularization](#regularization)
      - [L1 Regularization (Lasso)](#l1-regularization-lasso)
      - [L2 Regularization (Ridge)](#l2-regularization-ridge)
      - [Elastic Net Regularization](#elastic-net-regularization)
  - [Convergence Criteria](#convergence-criteria)
  - [Stopping Conditions](#stopping-conditions)
  - [Limitations](#limitations)
  - [Installation \& Usage](#installation--usage)
    - [Prerequisites](#prerequisites)
    - [Environment Setup](#environment-setup)
  - [License](#license)

---

## Motivation

In Machine Learning, we often create functions for predicting labels on new data. Now, our main task here is the 'training' stage, i.e. finding the optimal parameters for our model. Of course, there are also the data collection & preparation, validation testing, and deployment stages, which in and of themselves are interesting, but they are not the topic for today.

Today, we will be discussing Gradient Descent. Imagine we are trying to optimize some [**loss**](https://github.com/intelligent-username/Loss-Functions) function for a given set of data. Now, this can be done deterministically in a 'perfect' way in cases where a trivial analytical solution exists (for example, in the cases of [**linear regression**](https://github.com/intelligent-username/Linear-Regression) or even [**simple polynomial**](https://github.com/intelligent-username/Polynomial-Regression) regression). However, this has a few issues.

Firstly, in polynomial regression, if we try to fit the graph with an extremly high degree polynomial, we will get near-perfect accuracy on the training data, but the produced model will collapse under slight deviations in new or unseen data. This is called overfitting. To prevent this, we use regularization techniques, which are more easily applied in the context of gradient descent.

Next, there may not *be* a straight-forward analytic solution. If the loss surface is high-dimensional and not convex, or if the loss surface isn't differentiable *everywhere*, there won't be a simple `.solve()` function that can find a solution. Gradient descent, which is inspired by Newton's method, can instead iteratively approximate a decent solution.

Finally, there is the most practical concern: computational efficiency. Even if an analytic solution exists, it may be computationally expensive to compute directly. Consider the case where we have 10 dimensions and 3 million data points. The matrices involved in a direct solution could be humungous. Gradient descent, by contrast, can work with mini-batches of data and update parameters incrementally, making it more scalable for large datasets.

In this writeup, we will implement gradient descent from scratch and demonstrate it on regression problems, showing how iterative refinement can replace direct solutions in high‑dimensional, nonlinear cases.

---

## Math

Suppose you have some loss function $L(\theta)$ that you want to minimize with model parameters $\theta$.

The gradient descent algorithm works as follows:

$$
w_{i+1} = w_i - \eta \nabla L(w_i)
$$

Where:

- $w_i$ are the model parameters at iteration $i$ (e.g., weights in linear regression).
- $\eta$ is the *learning rate*, a small positive scalar that controls the step size. Note, in Newton's method, this is actually the inverse of the Hessian in place of the constant $\eta$, which is more accurate but way more costly.
- $\nabla L(w_i)$ is the gradient of the loss function at $w_i$, indicating the direction of steepest ascent (by the Cauchy-Schwarz Theorem).
- The negative sign, is inserted so we move from our current parameters closer to the minimum (also by the Cauchy-Schwarz Theorem).
- Finally, $w_{i+1}$ are the updated model parameters after iteration $i$.

This is the same as iterative minimization. We continue doing this until the change in loss, $\Delta L$ between iterations is smaller than some predefined threshold, the change in the gradient is below some threshold $\delta$, or until we run out of iterations. More formally, it is written with the condition:

$$
|L(w_{i+1}) - L(w_i)| < \epsilon
\\
\lor
\\
||\nabla L(w_{i+1})|| < \delta
\\
\lor
\\
i >= N
$$

(Where N is the maximum number of iterations, and $\epsilon$ and $\delta$ are small positive thresholds.)

Hence, the choice of learning rate is the only thing that can be tuned. The other parameters, if low enough, will affect the final result all that much. It is important to intelligently pick a learning rate: too small and the model will be slow (and simply never get close enough to the minimum), too large and the model will diverge (overshoot the minimum and oscillate out of control). The two most common strategies for fixing this are either fine-tuning a fixed learning rate after some trial-and-error, or using adjusting the learning rate dynamically during training (e.g., learning rate schedules or adaptive methods like Adam, more on that later).

Note that, often times, it is not enough to simply perform gradient descent. We will often times have separate training, validation, and testing sets. Also, we will have to pass through the data multiple times (epochs) to get a good fit. Finally, we will often times have to regularize the model to prevent overfitting.

---

## Variants

Now that we have the abstract gradient descent algorithm in general, we need to start implementing it specifically. There are three components that we need:

1) A [loss function](https://github.com/intelligent-username/Loss-Functions) $L(\theta)$ to minimize
2) A way to **compute** the gradient(s) $\nabla L(\theta)$, and
3) A strategy for updating the parameters (i.e., the learning rate $\eta$)

Today's topic will be the third components, as this is GD-specific.

### Batch Gradient Descent (full dataset)

This is the most 'vanilla' form of gradient descent. In this method, we have *some* loss function who's gradient changes depending on the parameters, $\theta$ that are passed into it.

We start off the gradient descent by choosing the initial parameters, $w_0$, either randomly or heuristically. For convex problems, initial parameters parameters can start anywhere because the Loss will converge anyway, but otherwise, they should be chosen carefully.

Then, we compute the gradient of the loss function at these parameters, $\nabla L(w_0)$:

$$
\nabla L(w_0) = \frac{1}{m} \sum_{i=1}^{m} \nabla_\theta l(f(x_i; w_0), y_i)
$$

Where:

- $m$ is the number of training examples
- $l$ is the individual loss function
- $f(x_i; w_0)$ is the model's prediction for input $x_i$ with parameters $w_0$
- and $y_i$ is the true label for input $x_i$.

We substitute the values of the initial parameters (a.k.a. the current point) into the gradient function to find the direction of greatest ascent.

Finally, we take a step in the opposite direction of the gradient, scaled by the learning rate.

We repeat this process with the newly attained parameter until one of the [stopping conditions](#stopping-conditions) is met.

Since Batch Gradient Descent computes the entire dataset through the gradient, it is very stable (a single update uses an entire epoch). However, it is also relatively slow. Oftentimes, it's even slower than finding the analytic solution (if it exists).

### Stochastic Gradient Descent

Stochastic Gradient Descent is identical to Batch Gradient Descent mathematically, but, instead of computing the gradient over the entire dataset, which takes a total of $O(m)$ per update, we compute it over a single data point (randomly selected (hence the *stochasticity*) without replacement). This dramatically reduces the computation time to $O(1)$.
The model converges a lot more quickly in terms of raw time, but takes more iterations to do so. Because we need more iterations, the batch count will also increase. Also, since we are sampling randomly, we will likely have a lot of noise, which makes the fit the final model weaker. This is actually a good thing, since it curbs overfitting. However, the noise can also make convergence more erratic, and it may oscillate around the minimum rather than settling down.

### Mini-Batch Gradient Descent

Mini-batch Gradient Descent is a compromise between Batch and Stochastic Gradient Descent. Instead of using the entire dataset or a single data point, we use a small, randomly selected subset of the data (a **mini-batch**) to compute the gradient. This allows us to take advantage of the stability of Batch Gradient Descent while still benefiting from the speed and robustness of Stochastic Gradient Descent.

Mini-batch gradient descent introduces a new hyperparameter which can be tuned: the batch size. Call this batch size $s$, with $1 < s < m$.
Each update will take $O(s)$ time, and create $O(m/s)$ updates per epoch. This is often the best balance between accuracy, speed, and generalization.

---

## Additional Techniques

The above variations use the same update rule but change how the loss's gradient is calculated by changing the batch size. The following techniques change the update rule itself. They can be used in tandem with any of the above batch size strategies to create beautiful, hybrid algorithms.

---

### Learning Rates

The learning rate, $\eta$, controls how big of a step we take in the direction of the negative gradient during each update. There are three main types of learning rates.

#### 1. Fixed

The simplest option is to set $\eta$ to some empirically-derived constant that seems to generally work well. However, this is not always optimal, as there is no real way to find an optimized universal learning rate. If the learning rate happens to be too high, the model will never converge, and, if it's too low, it will take too long. Often, experimenting with learning rates takes so much trial-and-error that it's not worth the effort.

#### 2. Scheduled

Scheduled learning rates start with some initial learning rate $\eta_0$ and then decay it over time according to some schedule. Common schedules include:

- **Step Decay**: Reduce the learning rate by a some scalar factor $\gamma \in (0,1)$ every $k$ epochs.
  
  $$
  \eta_k = \eta_0 \cdot \gamma^{\lfloor t / k \rfloor}
  $$

The constant $t$ sets the decay time scale.

- **Exponential Decay**: Continuously decay the learning rate exponentially.

  $$
  \eta_k = \eta_0 \cdot \gamma^{-\lambda k}
  $$

- **Inverse Time Decay**: Decay the learning rate inversely proportionally to the epoch number.
  $$
  \eta_k = \frac{\eta_0}{1 + \gamma k}
  $$

Scheduled rates/formulas also require empirical tuning for $\gamma$ and $\lambda$, but they are more consistent and convergent as loss functions tend to flatten out near a minimum anyway.

#### 3. Adaptive

Now, these are the real optimizations. Adaptive learning rates adjust $\eta$ based on the current behaviour of the loss function's gradient. This means that, with a 'strange' enough loss surface, the learning rate can go down, then back up, etc. and continue oscillating until convergence is reached.

##### Newton's Method

Based on Newton's root-finding technique, this method uses the second-order derivative (Hessian) to find the optimal learning rate. It can converge faster than first-order methods but is computationally expensive. It is written as:

$$
\eta_{i+1} = \eta_i - H^{-1} \nabla f(x_i)
$$

If we want to dampen the effects of the Hessian, we can add a decay term, $\lambda$, that, for example, updates after every batch:

$$
\eta_{i+1} = \eta_i - (H+\lambda_k I)^{-1} \nabla f(x_i)
$$

Although this method is highly accurate and mathematically elegant, it tends to be avoided due to the sheer cost of calculating, inverting, and transposing Hessians.

##### Adagrad

In some cases, different parameters need to be adjusted at different rates. For example, in natural language processing, some words are more common than others. We want to adjust the learning rate for each parameter based on how frequently it is updated.

In Adagrad, the learning rate is adjusted in proportion with the momentum of the gradient of the loss function. The general formula is:

$$
\eta_{t+1} = \frac{\eta_t}{\sqrt{G_{ti}} + \epsilon}
$$

Where

- The new learning rate is $\eta_{t+1}$.
- $G_{ii}$ is the sum of the squares of the gradients w.r.t. parameter $i$ up to the iteration at time $t$.
- $\epsilon$ is a constant to prevent division by zero (or reduce the impact of gradients).
- $\eta_0$ is the initial learning rate.

The only problem with Adagrad is that it increases monotonically, which can lead to the gradient shrinking too quickly and staying small permanently, thereby preventing further learning.

##### RMSProp

RMSProp fixes the shortcomings of Adagrad by using the exponentially weighted moving average, $E\left[g^2\right]$, of the squared gradients instead of the generic sum of squares.

The formula is:

$$
\eta_{t+1} = \frac{\eta_t}{\sqrt{E\left[g^2\right]_t} + \epsilon}
$$

##### Adam

Adam's method is the culmination of all of these adjustment methods. It creates parameter-specific learning rates that are adapted based on the first and second moments of the gradients. It then makes use of our next technique, momentum, to smooth out the updates.

$$
\eta_{t+1} = (\eta_t \hat{m}_t) \cdot \frac{1}{\sqrt{\hat{v}_t} + \epsilon}
$$

Here, $m$ is the 'momentum' and $v$ is the 'velocity'. We multiply by the 'momentum' to move in the general direction of the change in gradients. Then, we divide by the 'velocity' in order to normalize the step size, scaling updates down for parameters with consistently large gradients and up for sparse or small-gradient parameters.

Velocity is also known as the second moment (the uncentered variance), and momentum is also known as the first moment (the mean).

Where:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1)g_t
$$

With $\beta_1$ controlling how much 'history' of momentum we retain.

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2)g_t^2
$$

With $\beta_2$ controlling how much 'history' of velocity we retain. The greater $\beta_2$ is, the 'smoother' the adjustments, as recent gradients become less significant. Note that this is the same technique as used in RMSProp.

$$
g_t = \nabla f(x_{t-1})
$$

Note that both $m_t$ and $v_t$ are initialized to zero vectors.

Sometimes, if we want to be extra efficient, we adjust the bias of the $m_t$ and $v_t$, as their initialization at zero skews them downwards, making them inefficient early on. In this case, everywhere we see the variables $m_t$ and $v_t$, we replace them with $\hat{m}_t$ and $\hat{v}_t$, respectively, with:

$$
m_t = \frac{m_t}{1 - \beta_1^t}
$$

and

$$
v_t = \frac{v_t}{1 - \beta_2^t}
$$

\*Note that this bias correction becomes negligible as $t$ increases. This is only a small optimization and is mostly only visible in the first few iterations, so it may be neglected for the sake of saving compute time.

Also, we may occasionally reset our two moments to zero. For example, if the parameters start to oscillate or get stuck, or if the learning rate changes too drastically.

---

### Momentum

#### Polyak Momentum

#### Nesterov Acceleration

---

### Regularization

#### L1 Regularization (Lasso)

#### L2 Regularization (Ridge)

#### Elastic Net Regularization

---

## Convergence Criteria

---

## Stopping Conditions

---

## Limitations

- Can still overfit w/ improper regularization
- Can get stuck at local minima

---

## Installation & Usage

### Prerequisites

- C++, etc.
- Compiler
- IDE

### Environment Setup

Coming soon yoooooooooooooooooooo

1. Clone the project

2. Install dependencies (?) (TBD)

3. Use like so (TBM)

## License

Distributed under the [MIT License](LICENSE).

---
