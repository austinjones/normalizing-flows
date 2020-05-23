import Foundation
import TensorFlow

//let tensorflow_python = Python.import("tensorflow")

public struct RankOneConvolution: Layer {
    public var u: Tensor<Float>
    public var vT: Tensor<Float>
    
    init(dim: Int) {
        // generate a random rotation matrix by creating a matrix with standard normals
        // then, taking the Q part of the QR decompisition
        
        self.u = Tensor<Float>(randomNormal: [dim, 1], standardDeviation: Tensor(1e-5))
        self.vT = Tensor<Float>(randomNormal: [1, dim], standardDeviation: Tensor(1e-5))
    }
    
    @differentiable
    public func callAsFunction(_ input: TensorDeterminant) -> TensorDeterminant {
        let vT_x = matmul(vT, input.val.expandingShape(at: [2]))
        let y = input.val + matmul(u, vT_x).squeezingShape(at: [2])

        let det = Tensor<Float>(1) + matmul(vT, u).flattened()
        return TensorDeterminant(from: input, to_val: y, log_det: log(det))
    }
    
    public func reverse(_ output: TensorDeterminant) -> TensorDeterminant {
        let one = Tensor<Float>(1);
        let vT_y = matmul(vT, output.val.expandingShape(at: [2]))
        let vT_u = matmul(vT, u)
        let u_vT_y = matmul(u, vT_y).squeezingShape(at: [2])
        
        let one_over_one_plus_vT_u = one / (one + vT_u)
        let val = output.val - one_over_one_plus_vT_u.flattened() * u_vT_y
        let det = one / (one + matmul(vT, u))
        
        return TensorDeterminant(from: output, to_val: val, log_det: log(det))
    }
}
