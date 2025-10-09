# Gradient Descent

🚧In Progress🚧

![Cover](cover.jpg)

---

## Map

- [Gradient Descent](#gradient-descent)
  - [Map](#map)
  - [Motivation](#motivation)
  - [Math](#math)
  - [Margins](#margins)
  - [Maintenance](#maintenance)
  - [Mandate](#mandate)

---

## Motivation

In Machine Learning, we often create functions for predicting labels on new data. Now, our main task here is the 'training' stage, i.e. finding the optimal parameters for our model. Of course, there are also the data collection & preparation, validation testing, and deployment stages, which in and of themselves are interesting, but they are not the topic for today.

Today, we will be discussing Gradient Descent. Imagine we are trying to optimize some [**loss**](https://github.com/intelligent-username/Loss-Functions) function for a given set of data. Now, this can be done deterministically in a 'perfect' way in cases where a trivial analytical solution exists (for example, in the cases of [**linear regression**](https://github.com/intelligent-username/Linear-Regression) or even [**simple polynomial**](https://github.com/intelligent-username/Polynomial-Regression) regression). However, this has a few issues.

Firstly, in polynomial regression, if we try to fit the graph with an extremly high degree polynomial, we will get near-perfect accuracy on the training data, but the produced model will collapse under slight deviations in new or unseen data. This is called overfitting. To prevent this, we use regularization techniques, which are more easily applied in the context of gradient descent.

Next, there may not *be* a straight-forward analytic solution. If the loss surface is high-dimensional and not convex, or if the loss surface isn't differentiable *everywhere*, there won't be a simple `.solve()` function that can find a solution. Gradient descent, which is inspired by Newton's method, can instead iteratively approximate a decent solution.

Finally, there is the most practical concern: computational efficiency. Even if an analytic solution exists, it may be computationally expensive to compute directly. Consider the case where we have 10 dimensions and 3 million data points. The matrices involved in a direct solution could be humungous. Gradient descent, by contrast, can work with mini-batches of data and update parameters incrementally, making it more scalable for large datasets.

In this writeup, we will implement gradient descent from scratch and demonstrate it on regression problems, showing how iterative refinement can replace direct solutions in high‑dimensional, nonlinear cases.

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

We continue doing this until the change in loss, $\Delta L$ between iterations is smaller than some predefined threshold, the change in the gradient is below some threshold $\delta$, or until we run out of iterations. More formally, it is written with the condition:

$$
|L(w_{i+1}) - L(w_i)| < \epsilon \lor ||\nabla L(w_{i+1})|| < \delta \lor i >= N
$$

(Where N is the maximum number of iterations, and $\epsilon$ and $\delta$ are small positive thresholds.)

Here, the choice of learning rate is the only thing that can be tuned. The other parameters, if low enough, will affect the final result all that much. It is important to intelligently pick a learning rate: too small and the model will be slow (and simply never get close enough to the minimum), too large and the model will diverge (overshoot the minimum and oscillate out of control). The two most common strategies for fixing this are either fine-tuning a fixed learning rate after some trial-and-error, or using adjusting the learning rate dynamically during training (e.g., learning rate schedules or adaptive methods like Adam, more on that later).

## Margins

- Can still overfit w/ improper regularization
- Can get stuck at local minima

## Maintenance

(Download & running instructions coming soon)

## Mandate

Distributed under the [MIT License](LICENSE).

---
