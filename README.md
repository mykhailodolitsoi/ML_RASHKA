# Machine Learning with PyTorch and Scikit-Learn | S. Rashka | SUMMARY NOTES

1. [CH1-Machine Learning Types](#ch1)
2. [CH2-Perceptron](#ch2)

# CH1
Machine Learning is divided into three main types:
- Supervised Learning
- Unsupervised Learning
- Reinforcement Learning

---

## 1. Supervised Learning
Supervised learning is used when we have input features `X` and corresponding targets (labels) `y`.  
The goal is to learn a mapping function from inputs to outputs:

X -> y

### Main tasks:

#### Classification
Classification is the task of assigning a data point to a specific class based on labeled training data.  
Examples:
- spam / not spam detection
- image classification (cat vs dog)

#### Regression
Regression is the task of predicting continuous numerical values.  
Examples:
- house price prediction
- temperature prediction

---

## 2. Unsupervised Learning
Unsupervised learning is used when we only have input features `X` without labels.  
The goal is to discover hidden structure or patterns in the data.

### Main tasks:

#### Clustering
Clustering is the task of grouping similar data points together.  
Examples:
- customer segmentation

#### Dimensionality Reduction
Reducing the number of features while preserving important information.  
Used for:
- data compression
- visualization (e.g., PCA)

---

## 3. Reinforcement Learning
Reinforcement learning is based on an agent interacting with an environment.  

At each step:
- the agent observes a state
- takes an action
- receives a reward

The goal is to learn a policy (strategy) that maximizes cumulative reward over time.

Examples:
- game-playing agents (e.g., AlphaGo)
- autonomous driving
- robotics

# CH2
## Perceptron
Perceptron is a simple linear classifier.

### Model

z = w · x + b

Predicted output:

- y_pred = 1, if z >= 0
- y_pred = 0, if z < 0

where:
- x — input vector
- w — weight vector
- b — bias

---

### One Training Step

Given:
- x = (1, 1, 2)
- y = 0
- w = (1, 1, 1)
- b = -4
- learning rate = 0.1

---

### 1. Forward pass

z = 1 * 1 + 1 * 1 + 1 * 2 - 4 = 0

Since z >= 0, then:

- y_pred = 1

---

### 2. Error

error = y - y_pred = 0 - 1 = -1

---

### 3. Update rule

delta_w = learning_rate * error * x

delta_w = 0.1 * (-1) * (1, 1, 2) = (-0.1, -0.1, -0.2)

New weights:

w_new = w + delta_w  
w_new = (1, 1, 1) + (-0.1, -0.1, -0.2)  
w_new = (0.9, 0.9, 0.8)

New bias:

delta_b = learning_rate * error = 0.1 * (-1) = -0.1

b_new = b + delta_b = -4 + (-0.1) = -4.1

---

### Result

After one update:
- w = (0.9, 0.9, 0.8)
- b = -4.1

The perceptron changed its parameters because the prediction was wrong.

## Adaline
### Model

z = w · x + b

Predicted output:

- y_pred = z   (NO step function)

where:
- x — input vector
- w — weight vector
- b — bias

---

### Loss Function

L = (y - y_pred)^2

---

### One Training Step

Given:
- x = (1, 1, 2)
- y = 10
- w = (1, 1, 1)
- b = 1
- learning rate = 0.1

---

### 1. Forward pass

z = 1 * 1 + 1 * 1 + 1 * 2 + 1 = 5

y_pred = 5

---

### 2. Error

error = y - y_pred = 10 - 5 = 5

---

### 3. Gradient (from MSE)

delta_w = -2 * error * x

delta_w = -2 * 5 * (1, 1, 2) = (-10, -10, -20)

delta_b = -2 * error = -10

---

### 4. Update rule (gradient descent)

w_new = w - learning_rate * delta_w

w_new = (1, 1, 1) - 0.1 * (-10, -10, -20)  
w_new = (1 + 1, 1 + 1, 1 + 2)  
w_new = (2, 2, 3)

b_new = b - learning_rate * delta_b

b_new = 1 - 0.1 * (-10) = 2

---

### Result

After one update:
- w = (2, 2, 3)
- b = 2

The model updated parameters to reduce squared error.
