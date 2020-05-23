import Foundation
import TensorFlow

public struct HouseholderReflectionLayer: Layer {
    public var v: Tensor<Float>
    public var t: Tensor<Float>
    
    init(dim: Int) {
        self.v = Tensor(randomNormal: [dim, 1])
        self.t = Tensor(1.0)
    }
    
    @differentiable
    public func callAsFunction(_ input: TensorDeterminant) -> TensorDeterminant {
        // we need to divide by the l2_norm squared, so the sqrt and squared cancel
        let v_norm = v / sqrt(v.squared().sum())
        let v_vT_x = matmul(v_norm, vT_x)
        let val = input.val - (1.0 + t) * v_vT_x.squeezingShape(at: [2])
        
        return TensorDeterminant(from: input, to_val: val, log_det: Tensor<Float>(log(abs(t))))
    }
    
    public func reverse(_ output: TensorDeterminant) -> TensorDeterminant {
        let v_norm = v / sqrt(v.squared().sum())
        
        let vT_x = matmul(v_norm.transposed(), output.val.expandingShape(at: [2]))
        let v_vT_x = matmul(v_norm, vT_x)
        let x = output.val - (1.0 + t) * v_vT_x.squeezingShape(at: [2])
        
        return TensorDeterminant(from: output, to_val: x, log_det: Tensor(log(abs(t))))
    }
}
