from __future__ import annotations

from collections.abc import Mapping

from core.pipelines.base_pipeline import ResultCallback
from core.pipelines.deepstream.metadata_parser import parse_batch_meta


class ProbeManager:
    def __init__(
        self,
        callback: ResultCallback | None,
        source_ids: Mapping[int, str] | None = None,
        labels: Mapping[int, str] | None = None,
    ):
        self.callback = callback
        self.source_ids = dict(source_ids or {})
        self.labels = dict(labels or {})

    def attach(self, element: object, pad_name: str = "src") -> None:
        Gst = _require_gstreamer()
        pad = element.get_static_pad(pad_name)
        if pad is None:
            raise RuntimeError(
                f"Could not attach DeepStream metadata probe to pad '{pad_name}'."
            )
        pad.add_probe(Gst.PadProbeType.BUFFER, self._handle_buffer)

    def _handle_buffer(self, _pad, info):
        Gst = _require_gstreamer()
        if self.callback is None:
            return Gst.PadProbeReturn.OK

        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK

        pyds = _require_pyds()
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK

        for result in parse_batch_meta(
            batch_meta,
            source_ids=self.source_ids,
            labels=self.labels,
        ):
            self.callback(result)

        return Gst.PadProbeReturn.OK


def _require_gstreamer():
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        return Gst
    except (ImportError, ValueError) as error:
        raise RuntimeError("GStreamer is required for DeepStream probes.") from error


def _require_pyds():
    try:
        import pyds
    except ImportError as error:
        raise RuntimeError("pyds is required for DeepStream metadata probes.") from error
    return pyds
