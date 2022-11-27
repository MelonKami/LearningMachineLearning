import numpy as np

inputs = [1, 2, 3, 2.5] # input values to layer (this could be pixel values if you want image detection for example)
weights = [ [0.2, 0.8, -0.5, 1.0],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87]] # random weights set to tweak the output to the next layer (mutation)

biases = [2, 3, 0.5] # Bias is a value required for an activation (I believe)

output = np.dot(weights, inputs) + biases
print(output)
