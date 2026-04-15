# Machine Learning with PyTorch and Scikit-Learn | S. Rashka | SUMMARY NOTES

1. [CH1-Machine Learning Types](#ch1-machine-learning-types)
2. [CH2-Perceptron](#ch2-perceptron)

# CH1-Machine Learning Types
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

# CH2-Perceptron
