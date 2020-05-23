import Foundation
import TensorFlow
import PythonKit

let DATASETS = "machine-learning/datasets"
func load_and_train(fromCSV: String, epochs: Int = 10, batchSize: Int = 2000, depth: Int = 16, width: Int = 3) -> Model {
    let dataset = load_dataset(path: fromCSV, width: width)
    var model = load_model(dataset: dataset, depth: depth)
    print(model)
    train_model(model: &model, dataset: dataset.batched(batchSize), epochs: epochs)
    
    return model
}

func generate_inputs(model: Model, n: Int = 1000, width: Int = 3) -> Tensor<Float> {
    let generated_outputs = Tensor<Float>(randomNormal: [n, width])
    let with_det = TensorDeterminant(fromTensor: generated_outputs)
    return model.reverse(with_det).val
}

func save_csv(tensor: Tensor<Float>, toPath: String) {
    let np = Python.import("numpy")
    np.savetxt(toPath, tensor.makeNumpyArray(), delimiter: ",", fmt: "%0.10f")
}

let epochs = 50

let model = load_and_train(fromCSV: DATASETS + "/dataset/input.csv", epochs: epochs)
let generated = generate_inputs(model: model);
save_csv(tensor: generated, toPath: DATASETS + "/dataset/output-epoch-" + String(epochs) + ".csv")
