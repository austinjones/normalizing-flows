import Foundation
import TensorFlow
import PythonKit

struct Model: Layer {
    public var layers: [GlowLayer]
    
    init(batch: Tensor<Float>, depth: Int = 16) {
        let dim = batch.shape.dimensions[1];
        self.layers = []
        layers.append(GlowLayer(dataset: batch, normalize: true))
        
        for _ in 1..<depth {
            layers.append(GlowLayer(dim: dim))
        }
    }
    
    init(layers: [GlowLayer]) {
        self.layers = layers
    }
    
    init(join: [Model]) {
        self.layers = join.flatMap { m in m.layers.makeIterator() }
    }
    
    @differentiable
    public func callAsFunction(_ input: TensorDeterminant) -> TensorDeterminant {
        return layers(input)
    }
    
    public func intoChunks(size: Int) -> ChunkedModel {
        let models = self.layers.chunked(into: size)
            .map { chunk in Model(layers: chunk)}
        
        return ChunkedModel(models: models)
    }

    public func reverse(_ output: TensorDeterminant) -> TensorDeterminant {
        var output_with_det = output
        
        for layer in layers.makeIterator().reversed() {
            output_with_det = layer.reverse(output_with_det)
        }
        
        return output_with_det
    }
}

struct ChunkedModel: Layer {
    public var chunks: [Model]
    
    init(models: [Model]) {
        self.chunks = models
    }
    
    @differentiable
    public func callAsFunction(_ input: TensorDeterminant) -> TensorDeterminant {
        return self.chunks(input)
    }
    
    public func reverse(_ output: TensorDeterminant) -> TensorDeterminant {
        var output_with_det = output
        
        for model in self.chunks.makeIterator().reversed() {
            output_with_det = model.reverse(output_with_det)
        }
        
        return output_with_det
    }
}

public struct TensorDeterminant: Differentiable {
    public var val: Tensor<Float>
    public var log_det: Tensor<Float>
    
    @differentiable
    init(fromTensor: Tensor<Float>) {
        self.val = fromTensor
        self.log_det = Tensor<Float>(zeros: [fromTensor.shape.dimensions[0]])
    }
    
    @differentiable
    init(from: TensorDeterminant, to_val: Tensor<Float>, log_det: Tensor<Float>) {
        self.val = to_val
        self.log_det = from.log_det + log_det
    }
    
    @differentiable
    init(val: Tensor<Float>, log_det: Tensor<Float>) {
        self.val = val
        self.log_det = log_det
    }
}

extension Array: Module where Element: Layer, Element.Input == Element.Output {
    public typealias Input = Element.Input
    public typealias Output = Element.Output

    @differentiable(wrt: (self, input))
    public func callAsFunction(_ input: Input) -> Output {
          return self.differentiableReduce(input) { $1($0) }
    }
}

extension Array: Layer where Element: Layer, Element.Input == Element.Output {}

extension Array {
    func chunked(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0 ..< Swift.min($0 + size, count)])
        }
    }
}

extension Layer {
    mutating public func loadWeights(numpyFile: String) {
        let np = Python.import("numpy")
        let weights = np.load(numpyFile, allow_pickle: true)

        for (index, kp) in self.recursivelyAllWritableKeyPaths(to:  Tensor<Float>.self).enumerated() {
            self[keyPath: kp] = Tensor<Float>(numpy: weights[index])!
        }
    }

    public func saveWeights(numpyFile: String) {
        var weights: Array<PythonObject> = []

        for kp in self.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
            weights.append(self[keyPath: kp].makeNumpyArray())
        }

        let np = Python.import("numpy")
        np.save(numpyFile, np.array(weights))
    }
}
