# Vectorisation_Time_Savings

In neural networks with large numbers of input variables and therefore large numbers of gradients (for which we apply gradient descent to minimise the loss function), using a for loop to iterate over each variable-weight combination (i.e the activation function) will decrease the efficiency of the network as the number of inputs and layers increases, which is an implicit objective of deep learning models.

This exercise is concerned with an alternative technique to basic for loops, namely "Vectorisation", which makes use of Numpy's dot product method, and alleviates the need for a resource-consuming for loop.

The time saved is often more than a factor of 300.

```
import time

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a,b)
toc = time.time()

print("Output: " + str(c))
print("Vectorised version:" + str(1000*(toc-tic)) +"ms")

c = 0 
tic = time.time()
for i in range(1000000):
    c += a[i]*b[i]
toc = time.time()

print("\n","Output: " + str(c))
print("For loop version:" + str(1000*(toc-tic)) + "ms")
```
Example result:

Output: 250401.77777988912
Vectorised version:1.2679100036621094ms

 Output: 250401.7777798921
For loop version:443.05992126464844ms
