# Model Compression, C.Bucila
*Model Compression* trains ensembles of neural network on the original dataset, and uses these ensembles to label a large unlabeled dataset. Instead of training the neural network on the original training dataset,  *Model Compression* would train the neural network on this much larger, ensemble labeled dataset. This neural network can perform much better than a neural network with the same structure but  trained only using the original dataset.

---

# Distilling the Knowledge in a Neural Network, Hinton 
Deeper network tends to perform better than shallow ones and have stronger ability in abstracting information from datasets. Instead of training shallow network only using original datasets, we can train them to mimic the logits of deep networks. 

---

# Distilling Knowledge to Specialist Networks for Clustered Classification
Instead of just using a single deep network, we can train an ensembles of shallow networks to mimic the logits of the deep network, thus we can speed up 70 times in training and forward-passing than the original deep network. In these ensembles of shallow networks, we have two types of networks. One is the specialized network, which is responsible for a single set of related labels. Above these specialized networks, we have a network to summarize the logits of specialized network and produce the final result.

---
# FitNet: Hints for Thin Deep Nets
The most common method to achieve high energy efficiency is to reduce the size of network. *FitNet* uses deepper but thinner networks to mimic the original model, through which *FitNet* can use less parameters and computation to achieve similar accuracy compared with the original model. 

---

# Learning Efficient Object Detection Models with Knowledge Distillation
Implemented distillation on multiple networks and benchmarks and proved the effectiveness of distillation.

---

# MCDNN
Used a series of compression and specialization to generate sequence of variants of the original model, which generates the trade between accuracy and memory/energy. Then MCDNN uses an optimization model to choose variants according to current energy and memory budgets. 

---

# NoScope
