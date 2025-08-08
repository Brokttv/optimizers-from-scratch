# optimizers-from-scratch
Implementations of deep learning optimizers using NumPy, including SGD, Adam, Adagrad, NAG, RMSProp, and Momentum.
# Optimizers From Scratch in NumPy

This repository contains implementations of popular optimization algorithms using NumPy.  
Each optimizer is explained intuitively with the corresponding mathematical formulas.

---

## Optimizers

### 1. Stochastic Gradient Descent (SGD)

**Update rule:**

$$
\theta_{t+1} = \theta_t - \eta \cdot \nabla_{\theta} J(\theta_t)
$$

**Explanation:**  
Updates parameters by stepping in the opposite direction of the gradient computed on a mini-batch. Simple but sensitive to the choice of learning rate.

**Important Note:**  
SGD, full-batch gradient descent, and mini-batch gradient descent are conceptually the same algorithm but differ in batch size:  
1. **SGD**: batch size = 1 (updates per single sample)  
2. **Full-batch GD**: batch size = number of training samples (updates after full dataset pass)  
3. **Mini-batch GD**: batch size = chosen mini-batch (e.g., 2, 32, 64, etc.)  
Each setting affects noise and convergence speed.

**Further Reading:**  
🔗 [CS231n Optimization Overview (Stanford)](https://cs231n.github.io/optimization-1/)

---

### 2. Momentum

**Update rule:**

$$
v_t = \gamma v_{t-1} + \eta \cdot \nabla_{\theta} J(\theta_t)
$$

$$
\theta_{t+1} = \theta_t - v_t
$$

**Explanation:**  
Adds velocity to the update, accumulating past gradients to smooth out oscillations and speed up convergence.

**Further Reading:**  
🔗 [Distill.pub — Visualizing Momentum](https://distill.pub/2017/momentum/)

---

### 3. Nesterov Accelerated Gradient (NAG)

**Update rule:**

$$
v_t = \gamma v_{t-1} + \eta \cdot \nabla_{\theta} J(\theta_t - \gamma v_{t-1})
$$

$$
\theta_{t+1} = \theta_t - v_t
$$

**Explanation:**  
Improves upon Momentum by calculating the gradient at the projected future position, allowing for more responsive updates.

**Further Reading:**  
🔗 [Gradient Descent With Nesterov Momentum From Scratch — Machine Learning Mastery](https://machinelearningmastery.com/gradient-descent-with-nesterov-momentum-from-scratch/)

---

### 4. AdaGrad

**Update rule:**

$$
r_t = r_{t-1} + \nabla_{\theta} J(\theta_t)^2
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{r_t} + \epsilon} \cdot \nabla_{\theta} J(\theta_t)
$$

**Explanation:**  
Adapts learning rates per parameter based on the cumulative sum of squared gradients, favoring infrequent parameters. Learning rates shrink over time.

**Further Reading:**  
🔗 [An overview of gradient descent optimization algorithms — Ruder](https://ruder.io/optimizing-gradient-descent/index.html#adagrad) — one of the clearest breakdowns on the web, from a well-respected ML researcher.

---

### 5. RMSProp

**Update rule:**

$$
r_t = \rho r_{t-1} + (1 - \rho) \cdot \nabla_{\theta} J(\theta_t)^2
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{r_t} + \epsilon} \cdot \nabla_{\theta} J(\theta_t)
$$

**Explanation:**  
Uses an exponentially weighted moving average of squared gradients to fix AdaGrad's rapid decay of learning rates, stabilizing training.

**Further Reading:**  
🔗 [Geoff Hinton's Coursera Lecture — RMSProp](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)


---

### 6. Adam (Adaptive Moment Estimation)

**Update rule:**

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) \cdot \nabla_{\theta} J(\theta_t)
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) \cdot \nabla_{\theta} J(\theta_t)^2
$$

Bias correction:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

Final update:

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \cdot \hat{m}_t
$$

**Explanation:**  
Combines Momentum and RMSProp by keeping track of both the first moment (mean) and second moment (variance) of gradients. It adapts learning rates per parameter and smooths updates, making it very effective for many tasks.

**Further Reading:**  
🔗 [Adam Paper — "Adam: A Method for Stochastic Optimization"](https://arxiv.org/abs/1412.6980)


---

## Usage

- Each optimizer is implemented in a clear and commented notebook cell.  
- Experiment with hyperparameters like learning rate and momentum factor for best results.




