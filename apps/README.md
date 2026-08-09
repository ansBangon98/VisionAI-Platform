# Application Pipelines

`configs/apps/*.yaml` files define application variants. The YAML filename is
only the variant key shown in the UI.

The Python implementation is selected by:

```yaml
application:
  pipeline: detection
```

For example, `people_analytics.yaml`, `demo_detection.yaml`, and
`vehicle_analytics.yaml` can all reuse `apps/detection/pipeline.py`.

