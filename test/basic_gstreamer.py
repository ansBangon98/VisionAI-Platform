import sys

import gi

gi.require_version("Gst", "1.0")

from gi.repository import GLib, Gst


def main() -> int:
    Gst.init(None)

    try:
        pipeline = Gst.parse_launch(
            "videotestsrc is-live=true pattern=ball "
            "! videoconvert "
            "! autovideosink"
        )
    except GLib.Error as error:
        print(f"Failed to create pipeline: {error}", file=sys.stderr)
        return 1

    bus = pipeline.get_bus()
    bus.add_signal_watch()

    main_loop = GLib.MainLoop()

    def on_message(_bus: Gst.Bus, message: Gst.Message) -> None:
        if message.type == Gst.MessageType.ERROR:
            error, debug_info = message.parse_error()
            print(f"GStreamer error: {error}", file=sys.stderr)

            if debug_info:
                print(f"Debug information: {debug_info}", file=sys.stderr)

            main_loop.quit()

        elif message.type == Gst.MessageType.EOS:
            print("End of stream")
            main_loop.quit()

    bus.connect("message", on_message)

    result = pipeline.set_state(Gst.State.PLAYING)

    if result == Gst.StateChangeReturn.FAILURE:
        print("Failed to start the GStreamer pipeline.", file=sys.stderr)
        pipeline.set_state(Gst.State.NULL)
        return 1

    print("GStreamer is running. Press Ctrl+C to stop.")

    try:
        main_loop.run()
    except KeyboardInterrupt:
        print("\nStopping pipeline...")
    finally:
        pipeline.set_state(Gst.State.NULL)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())