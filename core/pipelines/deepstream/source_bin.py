from __future__ import annotations

from pathlib import Path


def create_source_bin(index: int, uri: str):
    Gst = _require_gstreamer()
    bin_name = f"source-bin-{index:02d}"
    source_bin = Gst.Bin.new(bin_name)
    if source_bin is None:
        raise RuntimeError(f"Unable to create DeepStream source bin {bin_name}.")

    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", f"uri-decode-bin-{index}")
    if uri_decode_bin is None:
        raise RuntimeError("Missing GStreamer element: uridecodebin")

    uri_decode_bin.set_property("uri", uri)
    uri_decode_bin.connect("pad-added", _handle_decodebin_pad_added, source_bin)
    uri_decode_bin.connect("child-added", _handle_decodebin_child_added)

    source_bin.add(uri_decode_bin)
    ghost_pad = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
    if ghost_pad is None or not source_bin.add_pad(ghost_pad):
        raise RuntimeError("Unable to add DeepStream source ghost pad.")

    return source_bin


def source_uri_from_config(config: dict) -> str:
    source_type = str(config.get("type", "")).strip().lower()
    if source_type == "rtsp":
        uri = str(config.get("uri", "")).strip()
        if not uri:
            raise RuntimeError("DeepStream RTSP source requires a resolved uri.")
        return uri

    if source_type in {"video_file", "file", "video"}:
        path = str(config.get("path", "")).strip()
        if not path:
            raise RuntimeError("DeepStream video file source requires path.")
        if "://" in path:
            return path
        return Path(path).expanduser().resolve().as_uri()

    raise RuntimeError(
        f"DeepStream source type '{source_type}' is not supported yet. "
        "Use an RTSP or video file source."
    )


def _handle_decodebin_pad_added(decodebin, decoder_src_pad, source_bin) -> None:
    caps = decoder_src_pad.get_current_caps()
    if caps is None:
        caps = decoder_src_pad.query_caps()
    structure = caps.get_structure(0)
    name = structure.get_name()
    if "video" not in name:
        return

    features = caps.get_features(0)
    if not features.contains("memory:NVMM"):
        raise RuntimeError(
            "DeepStream decodebin selected a CPU decoder. NVIDIA NVMM memory is "
            "required for the DeepStream pipeline."
        )

    ghost_pad = source_bin.get_static_pad("src")
    if ghost_pad is None:
        raise RuntimeError("DeepStream source bin has no src ghost pad.")
    if ghost_pad.set_target(decoder_src_pad):
        return
    raise RuntimeError("Unable to link decodebin src pad to source ghost pad.")


def _handle_decodebin_child_added(_decodebin, child, _name) -> None:
    if child.find_property("drop-on-latency") is not None:
        child.set_property("drop-on-latency", True)


def _require_gstreamer():
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        return Gst
    except (ImportError, ValueError) as error:
        raise RuntimeError("GStreamer is required for DeepStream source bins.") from error
