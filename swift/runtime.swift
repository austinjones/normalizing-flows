import Foundation
import TensorFlow

func load(path: String, width: Int) -> Tensor<Float> {
    let url = URL(fileURLWithPath: path)
    
    let csv_data = try! String(
        contentsOf: url,
        encoding: String.Encoding.utf8);

    let parsed: [Float] = csv_data.split(separator: "\n").flatMap {
        String($0).split(separator: ",").compactMap { Float(String($0)) }
    }
    
    return Tensor<Float>(parsed).reshaped(to: [parsed.count / width, width])
}

func load_dataset(path: String, width: Int) -> Dataset<Tensor<Float>> {
    let tensor = load(path: path, width: width);
    return Dataset(elements: tensor)
}

func load_model(dataset: Dataset<Tensor<Float>>, depth: Int = 16) -> Model {
    let first_batch = get_first_batch(dataset: dataset, n: 1000)
    return Model(batch: first_batch, depth: depth)
}

func train_chunked_model(model: inout ChunkedModel, dataset: Dataset<Tensor<Float>>, epochs: Int, printStatus: Bool = false) {
    var optimizers: [Adam] = model.chunks.map { chunk in Adam(for: chunk, learningRate: 1e-3) }
    
    var batch_id = 0
    for _epoch in 1...epochs {
        for batch in dataset {
            var input = TensorDeterminant(fromTensor: batch)
            var e_px: [Float] = []
            
            for chunkId in 0..<model.chunks.count {
                let optimizer = optimizers[chunkId]
                var modelChunk = model.chunks[chunkId]
                
                let (value, grad) = TensorFlow.valueWithGradient(at: modelChunk) { modelChunk -> Tensor<Float> in
                    if printStatus {
                       print("batch #", batch_id)
                    }

                    let output = modelChunk(input)
                    let loss = normalNegLogLikelihoodLoss(data_det: output, printStatus: printStatus)

                    if printStatus {
                       print("NLL:    ", loss)
                       print("E[P x]: ", exp(-loss))
                    }

                    withoutDerivative(at: output, in: {o in input = o})
                    
                    return loss
                }
                
                e_px.append(exp(-value.scalar!))

                if printStatus {
                    print("")
                }

                optimizer.update(&modelChunk, along: grad)
                
                optimizers[chunkId] = optimizer
                model.chunks[chunkId] = modelChunk
                
                batch_id += 1
            }
            
            print("E[P x] @ chunk:", e_px)
        }
    }
}

func train_model(model: inout Model, dataset: Dataset<Tensor<Float>>, epochs: Int, printStatus: Bool = true) {
    let optimizer = Adam(for: model, learningRate: 1e-3)
    var batch_id = 0
    for _epoch in 1...epochs {
        for batch in dataset {
            var input = TensorDeterminant(fromTensor: batch)
                
            let grad = TensorFlow.gradient(at: model) { model -> Tensor<Float> in
                if printStatus {
                   print("batch #", batch_id)
                }

                let output = model(input)
                let loss = normalNegLogLikelihoodLoss(data_det: output, printStatus: printStatus)

                if printStatus {
                   print("NLL:    ", loss)
                   print("E[P x]: ", exp(-loss))
                }

                withoutDerivative(at: output, in: {x in input = x})
                
                return loss
            }

            print("")

            optimizer.update(&model, along: grad)
            batch_id += 1
        }
    }
}
