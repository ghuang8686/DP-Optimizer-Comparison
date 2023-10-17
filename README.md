# DP-Optimizer-Comparison
This project is to compare the common optimizers in Deep Learning

As we know, gradient descend is a very common optimizer in the optimization problems. 

When it is applied to numerical analysis, i.e., for the scatter data points, we can use the average gradient over all the data points as the updated gradient. When the size of the dataset is high, the computing is costly. 

One alternate solution is batch gradient descend (BGD). The idea is instead to average over the entire dataset, we can split the whole dateset into several subsets with the equal size. The size is usually the exponential with base 2, like 64, 128, 256 etc. In each iteration, we can simple use a single subset (batch) to calculate the gradient, until the error is satisfied. In general, the entire dataset will be utilized with enough epoches. This optimizer can utilize the feature of parallel computing of the device (Though I don't know the details). But it is memory cost as the entire dataset is stored as batches (no repeat or random data points) in the memory.

One variant of BGD is mini-batch gradient descent. In general, we will generate a smaller and random mini-batch from the dataset. It is more efficient in memory as monly one mini-batch needs to be held at a time. But not the entire data is guaranteed to be used. 

Another variant is stochastic gradient descend, which is even more extreme. A random data point is selected to update the gradient descent. It can converge faster because it updates the model more frequently and escape the local minima more efficiently due to the noisy update. But it has high variance and noise in parameter updates. And also the learning rate should be selected carefully. It also can be less computationly efficient on GPU.

In the original version of GD, the learning rate is fixed. But we want the learning rate decays over iteration. One variant is Adagrad (Adaptive Gradient Descent). It is very useful when dealing with sparse data or when different parameters have different scales. The limitation is that the learning rates tend to decrease over time, which can lead to very slow convergence.

One improvement is RMSprop, which stands for Root mean Square Propagation. Instead of the sum of the previous gradients, RMSprop uses a moving average of squared gradients to adapt the learning rates for each parameter individually. The moving average introduces a smoother decay factor, which allows the learning rate to adapt more effectively to the characteristics of the optimization landscape. This means that the learning rate can increase, stay relatively constant, or decrease more gradually depending on the historical gradients.

The superior optimizer is Adam (Adaptive moment estimation), which combines the Adagrad and RMSprop. The results show that Adam converges fastest among the optimizer after 100 iterations. 
