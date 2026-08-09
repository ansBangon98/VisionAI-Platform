# People Analytics

People Analytics is the application layer for counting and tracking people in the live camera feed.

## Current Features

- Runs the people/face detector from `models/detection/people_face`.
- Selects the detector runtime from `detector.backend` in `configs/apps/people_analytics.yaml`.
- Filters detections to the `person` class.
- Tracks people with ByteTrack from `core/tracking/bytetrack`.
- Returns only `track_id` and `bbox` from the pipeline.
- Appears in the main UI `cbo_selected_test` using `application.name` from `configs/apps/people_analytics.yaml`.
- Uses shared camera sources from `configs/cameras.yaml`.
- Updates the main demo UI object counter (`lbl_Objects`) with the current tracked person count.

## Pipeline Output

Each processed frame returns a list like:

```python
[
    {
        "track_id": 1,
        "bbox": (120, 84, 260, 420),
    }
]
```

## Planned Extensions

Future predictors such as gender, emotion, staff/customer classification, zone dwell time, heatmaps, restricted-zone alerts, and customer assistance status should be added in `analytics.py` or as explicit model stages after the base people tracker is stable.

## Detector Backends

The detector can use these backend values:

```yaml
application:
  pipeline: detection

primary_model:
  backend: onnxruntime
```

Supported values are `onnxruntime`, `openvino`, `pytorch`, and `tensorrt`.

The model file must match the selected runtime. ONNX Runtime uses `.onnx`; TensorRT uses a serialized engine such as `.engine`, `.plan`, or `.trt`.

Model artifacts live outside application code:

```text
models/detection/people_face/
  model.onnx
  model.xml
  model.bin
  labels.txt
  metadata.yaml
```

`metadata.yaml` is the preferred source for class names and model settings.
`labels.txt` stays beside the model for runtimes such as DeepStream.

## Camera Source

The app config only selects a camera by name:

```yaml
source:
  name: office_entrance
```

The actual camera definition is shared in `configs/cameras.yaml`. RTSP URLs should be stored in `.env` and referenced by `uri_env`.
