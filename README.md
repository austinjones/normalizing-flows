# Normalizing Flows
An exploration of Normalizing Flows, in Pytorch and Swift for Tensorflow.


## Pytorch
In the notebooks directory, I focused on developing a rigorous method of benchmarking flow performance 
and testing that flow implementations have correct log determinants, and inverses.  

### KL Divergence Benchmarking
One of my most interesting discoveries is that it is possible to benchmark performance based on the KL divergence 
between the data distribution and the normalized distribution (under the change of variables theorem).  This can be evaluated 
via Monte Carlo estimation, as `Mean [ log P(data) - log N(output) - log determinant ]`, where `P` is a chosen source distriution, 
`N` is the multivariate gaussian, and `log determinant` is an output of the flow model.

In fact, the RealNVP/Glow objective function is simply the second term of the KL divergence 
(as the first term is a constant under optimization).
If the flow performs perfectly, the KL divergence will reduce to 0.

This method is convenient when comparing performance across problems, since the KL divergence metric is 
an absolute measure of divergence and can be used to compare flow performance across source distributions.

It can also be evaluated at any layer of the network, not just the final layer 
(by assuming the output is gaussian, and using the log determinant up to that layer).

### Pytorch Flows
The [flows.py](https://github.com/austinjones/normalizing-flows/blob/master/notebooks/flows.py)
file has implementations of novel non-linear activation functions:
- [SoftlogFlow](https://github.com/austinjones/normalizing-flows/blob/master/notebooks/flows.py#L79), the elementwise function of y = sign(x) log(abs(x) + 1)
- [SoftsquareFlow](https://github.com/austinjones/normalizing-flows/blob/master/notebooks/flows.py#L100), a residual activation function, elementwise y = a * x + b * sign(x) * x^2

## Swift
In the swift directory, I implemented the flows described in RealNVP and Glow.  Of interest are:
- [AffineCoupling.swift](https://github.com/austinjones/normalizing-flows/blob/master/swift/FlowTransforms/AffineCoupling.swift), the affine coupling layer.
- [InvertibleConvolutionQr.swift](https://github.com/austinjones/normalizing-flows/blob/master/swift/FlowTransforms/InvertibleConvolutionQr.swift), an alternative method to invert a 1x1 convolution based on QR decomposition.
- [HouseholderReflection.swift](https://github.com/austinjones/normalizing-flows/blob/master/swift/FlowTransforms/HouseholderReflection.swift), an implementation of Householder Reflections about a trainable hyperplane.

### Real Data
I applied the models to an interesting problem: learning the distribution of colors in a palette.  
These datasets are easy to extract from images, and the distributions can be quirky, which challenges the models.

Here are colors sampled from the 'purple-forest' source dataset:
![training colors](https://raw.githubusercontent.com/austinjones/normalizing-flows/master/datasets/purple-forest.jpg)

These are colors which were generated from a normalizing flow model, with about 500 weights:
![generated colors](https://raw.githubusercontent.com/austinjones/normalizing-flows/master/datasets/purple-forest-epoch-50.jpg)
