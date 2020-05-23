import Foundation
import TensorFlow

public struct ActNormLayer: Layer {
    public var s: Tensor<Float>
    public var b: Tensor<Float>
    
    init(dataset: Tensor<Float>) {
        let stddev = dataset.standardDeviation(squeezingAxes: [0])
        let adjusted_stddev = dataset / stddev
        let mean = adjusted_stddev.mean(squeezingAxes: [0])
        
        self.s = 1.0 / stddev
        self.b = -1 * mean
    }
    
    init(shape: TensorShape) {
        self.s = Tensor(repeating: 1.0, shape: shape)
        self.b = Tensor(repeating: 0.0, shape: shape)
    }
    
    @differentiable
    public func callAsFunction(_ input: TensorDeterminant) -> TensorDeterminant {
        let val = s * input.val + b
        let log_det = log_determinant()
        
        return TensorDeterminant(from: input, to_val: val, log_det: log_det)
    }
    
    public func reverse(_ output: TensorDeterminant) -> TensorDeterminant {
        let val = (output.val - b) / s
        let log_det = 1.0 / log_determinant()
        
        return TensorDeterminant(from: output, to_val: val, log_det: log_det)
    }
    
    func log_determinant() -> Tensor<Float> {
        return log(abs(self.s)).sum()
    }
}
