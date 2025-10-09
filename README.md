# Gradient Descent From Scratch

## 🚧In Progress🚧

![Cover](img.jpg)

---

## Outline

Same Template as always :)

- [Gradient Descent From Scratch](#gradient-descent-from-scratch)
  - [🚧In Progress🚧](#in-progress)
  - [Outline](#outline)
  - [The Math Behind Gradient Descent](#the-math-behind-gradient-descent)
  - [Installation](#installation)
  - [API](#api)
  - [License](#license)

---

## The Math Behind Gradient Descent

Motivation & Theory
Note: for context, you might want to read up on Polynomial Regression first. Gradient descent builds directly on that foundation.

Say we have a loss function and a model whose parameters we want to tune. For simple cases like linear or polynomial regression, we can solve for the optimal parameters directly. However, most real‑world models—especially those with many parameters or nonlinear structure—are too complex for such a deterministic solution. Ecce Gradient Descent.

Gradient descent is an iterative optimization method that updates model parameters by stepping in the direction of steepest descent of the loss surface. This “follow the slope” approach allows us to find minima without needing a closed‑form solution. The step size is controlled by the learning rate, which balances speed of convergence against stability.

In this writeup, we will implement gradient descent from scratch and demonstrate it on regression problems, showing how iterative refinement can replace direct solutions in high‑dimensional, nonlinear cases.

## Installation

## API

## License

Distributed under the [MIT License](LICENSE).

---
