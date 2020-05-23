import Foundation
import TensorFlow
import PythonKit

// Box-Muller transform from uniform space, to normal space
func uniform_to_normal(uniform: Tensor<Float>) -> Tensor<Float> {
    let u1 = uniform;
    let u2 = Tensor<Float>(randomUniform: u1.shape)
    return sqrt(-2 * log(u1)) * cos(2 * Float.pi * u2)
}

// Box-Muller transform from normal space, to uniform space
func normal_to_uniform(normal: Tensor<Float>) -> Tensor<Float> {
    let z1 = normal;
    let z2 = Tensor<Float>(randomNormal: z1.shape)
    return exp((-1/2) * (z1.squared() + z2.squared()))
}

public func random_rotation(dim: Int) -> Tensor<Float> {
    let tensor = Tensor<Float>(
        randomNormal: [dim, dim],
        mean: Tensor<Float>(0.0),
        standardDeviation: Tensor<Float>(1.0)
    )
    
    let qr = tensor.qrDecomposition()
    return qr.q
}

public func slow_python_determinant(t: Tensor<Float>) -> Tensor<Float> {
    let tensorflow_python = Python.import("tensorflow")
    print("Invoking numpy for determinant")
    let det_numpy = tensorflow_python.linalg.det(t.makeNumpyArray())
        .numpy();
    let swift_val = Float(det_numpy)!
    return  Tensor<Float>(swift_val)
}

func get_first_batch(dataset: Dataset<Tensor<Float>>, n: Int) -> Tensor<Float> {
    var iter = dataset.batched(n).makeIterator()
    return iter.next()!
}

@differentiable
func normalNegLogLikelihoodLoss(data_det: TensorDeterminant, printStatus: Bool = true) -> Tensor<Float> {
    let data = data_det.val
    let det = data_det.log_det
    
    if printStatus {
        print("mean:   ", data.mean(squeezingAxes: [0]))
        print("stddev: ", data.standardDeviation(squeezingAxes: [0]))
    }
    
    let k = Float(data.shape.dimensions[1])
    let xt_x = data.squared().sum(alongAxes: [1]).squeezingShape(at: 1);

    let log_pdf = log(pow(2 * Float.pi, -k / 2)) - xt_x / 2
    let objective = log_pdf + det
    
    let mean: Tensor<Float> = -1.0 * objective.mean()
    return mean
}
