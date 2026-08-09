# Model Assets

Store deployable model files here. Keep Python detector/runtime code under
`core/models`.

Recommended layout:

```text
models/
  detection/<model_name>/metadata.yaml
  detection/<model_name>/labels.txt
  detection/<model_name>/model.onnx
  detection/<model_name>/model.xml
  detection/<model_name>/model.bin
  detection/<model_name>/model.engine
  classification/<model_name>/metadata.yaml
  segmentation/<model_name>/metadata.yaml
  embedding/<model_name>/metadata.yaml
  pose/<model_name>/metadata.yaml
```

`metadata.yaml` is the platform source of truth for model behavior. Keep
`labels.txt` next to the artifact for runtimes such as DeepStream that expect a
plain label file.
