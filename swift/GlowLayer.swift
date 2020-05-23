import Foundation
import TensorFlow

public struct GlowLayer: Layer {
    public var act_norm: ActNormLayer
    public var inv_conv: InvertibleConvolutionLayerQr
    public var affine_coupling: AffineCouplingLayer
        
    init(dataset: Tensor<Float>, normalize: Bool) {
        let dim = dataset.shape.dimensions[1];
        if normalize {
            self.act_norm = ActNormLayer(dataset: dataset)
        } else {
            self.act_norm = ActNormLayer(shape: [dim])
        }
        
        self.reflection = HouseholderReflectionLayer(dim: dim)
        self.rank_one_conv = RankOneConvolution(dim: dim)
        self.affine_coupling = AffineCouplingLayer(shape: dim)
    }
    
    init(dim: Int) {
        self.act_norm = ActNormLayer(shape: [dim])
        self.inv_conv = InvertibleConvolutionLayerQr(dim: dim)
        self.affine_coupling = AffineCouplingLayer(shape: dim)
    }
    
    @differentiable
    public func callAsFunction(_ input: TensorDeterminant) -> TensorDeterminant {
        return affine_coupling( inv_conv ( act_norm (input)))
    }
    
    public func reverse(_ output: TensorDeterminant) -> TensorDeterminant {
        return act_norm.reverse( rank_one_conv.reverse( affine_coupling.reverse(output)))
    }
}
