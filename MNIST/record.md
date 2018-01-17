# Repeat Hinton's Result on Distillation
## Results claimed by Hinton
| # of units per hiddent layer | Temperature | Dropout | # of errors | Accuracy |
| --- | --- |--- | --- | --- |
| 1200 | NA | Yes | 67 | 99.33% | 
|  800 | NA | No | 146 | 98.54% |
|  800 | 20 | No | 74 | 99.26% |


## Results produced by Boyuan
| # of units per hiddent layer | Temperature | Dropout | # of errors | Accuracy |
| --- | --- |--- | --- | --- |
| 1200 | NA | Yes | 171 | 98.29% | 
|  800 | NA | Yes | 177 | 98.23% |
|  300 | NA | Yes | 196 | 98.04% |
|  30 | NA | Yes | 648 | 93.52% |
|  800 | NA | No | 184 | 98.16% |
|  300 | NA | No | 217 | 97.83% |
|  30 | NA | No | 577 | 94.23% |
|  30 | 2 | No | 435 | 95.65% |
|  30 | 4 | No | 423 | 95.77% |
|  30 | 8 | No | 402 | 95.98% |
|  30 | 20 | No | 419 | 95.81% |
|  30 | 100 | No | 435 | 95.65% |
|  30 | 1000 | No | 459 | 95.41% |

---
Notes1: 
Claimed temperature by Hinton fror 30 units: 2.5 ~ 4.0

---
Notes2: 
Two hidden layer model with 1200  units per layer. Using GreatestGradientOptimizer, the accuray is 94%. Using AdamOptimizer, the accuracy is 98%.

---
