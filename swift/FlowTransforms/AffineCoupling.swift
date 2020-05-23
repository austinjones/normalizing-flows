import Foundation
import TensorFlow

public struct AffineCouplingNetwork: Layer {
    public var s1: Dense<Float>
    public var t1: Dense<Float>
    
    init(from: Int, inner: Int, to: Int) {
        // initialize the weights so the network returns s=1 and t=0,
        // which will result in an identity function for the AffineCouplingLayer
        self.s1 = Dense<Float>(
            inputSize: from,
            outputSize: to,
            weightInitializer: zeros(),
            biasInitializer: zeros()
        )

        self.t1 = Dense<Float>(
            inputSize: from,
            outputSize: to,
            weightInitializer: zeros(),
            biasInitializer: zeros()
        )
    }
    
    // Returns log_s, and t
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> [Tensor<Float>] {
        return [s1(input), t1(input)]
    }
}

public struct AffineCouplingLayer: Layer {
    public var nn: AffineCouplingNetwork
    
    @noDerivative
    var left_len: Int
    
    @noDerivative
    var right_len: Int
    
    @noDerivative
    var input_len: Int
    
    init(shape: Int) {
        self.left_len = Int.random(in: 1...(shape-1))
        self.right_len = shape - left_len
        self.input_len = shape
        
        let inner = max(left_len, right_len)
        
        self.nn = AffineCouplingNetwork(from: right_len, inner: inner, to: left_len)
    }
    
    @differentiable
    public func callAsFunction(_ input_det: TensorDeterminant) -> TensorDeterminant {
        let input = input_det.val
        if input.shape.dimensions[1] != input_len {
            print("Shape error!!")
        }
        
        let parts = input.split(sizes: Tensor<Int32>([Int32(left_len), Int32(right_len)]), alongAxis: 1)
        
        let x_left = parts[0]
        let x_right = parts[1]
        
        let network_result = nn(x_right)
        let log_s = network_result[0]
        let s = exp(log_s)
        let t = network_result[1]
        
        let y_left = s * x_left + t
        let y_right = x_right
        
        let val = Tensor<Float>(concatenating: [y_left, y_right], alongAxis: 1);
        let log_det = log_determinant(s: s)
        
        return TensorDeterminant(from: input_det, to_val: val, log_det: log_det)
    }
    
    public func reverse(_ output_det: TensorDeterminant) -> TensorDeterminant {
        let output = output_det.val
        let parts = output.split(sizes: Tensor<Int32>([Int32(left_len), Int32(right_len)]), alongAxis: 1);
        
        let y_left = parts[0]
        let y_right = parts[1]
        
        let network_result = nn(y_right)
        let log_s = network_result[0]
        let s = exp(log_s)
        let t = network_result[1]
        
        let x_left = (y_left - t) / s
        let x_right = y_right
        
        let val = Tensor<Float>(concatenating: [x_left, x_right], alongAxis: 1)
        let log_det = 1.0 / log_determinant(s: s)
        
        return TensorDeterminant(from: output_det, to_val: val, log_det: log_det)
    }
    
    func log_determinant(s: Tensor<Float>) -> Tensor<Float> {
        return log(abs(s)).sum(alongAxes: [1]).flattened()
    }
}
