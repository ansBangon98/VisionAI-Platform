import importlib
import os
import sys
import traceback
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSlider,
)
from PySide6.QtCore import QEvent, QFile, QObject, QThread, Signal, Slot
from PySide6.QtGui import QImage
from PySide6.QtUiTools import QUiLoader

from apps.registry import AppDefinition, discover_app_configs
from core.camera import CameraFactory, CameraRegistry, CameraSourceDefinition
from core.camera.camera_config import CameraMetrics
from core.pipelines.cpu.frame_processor import frame_result_to_legacy_objects
from core.pipelines.cpu.gstreamer_pipeline import attach_camera_viewer
from core.results.frame_result import FrameResult


try:
    import assets.icons.icons_rc
except ImportError:
    pass


# HELPER FUNCTIONS

def resource_path(relative: str) -> str:
    """PyInstaller-aware resource path."""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative)
    project_root = Path(__file__).resolve().parent.parent
    return str(project_root / relative)


class AnalyticsWorker(QObject):
    results_ready = Signal(object, object, object)
    error_occurred = Signal(str, str)
    processing_finished = Signal()

    def __init__(self, pipeline, analytics, app_name: str):
        super().__init__()
        self.pipeline = pipeline
        self.analytics = analytics
        self.app_name = app_name

    @Slot(QImage)
    def process_frame(self, frame: QImage):
        try:
            frame_result = self.process_frame_result(frame)
            if frame_result is not None:
                results = frame_result_to_legacy_objects(frame_result)
                summary = self.analytics.update(frame_result)
            else:
                results = self.pipeline.process(frame)
                summary = self.analytics.update(results)
            source_size = (frame.width(), frame.height())
            self.results_ready.emit(results, summary, source_size)
        except RuntimeError as error:
            self.error_occurred.emit(str(error), traceback.format_exc())
        except Exception as error:
            runtime_error = RuntimeError(
                f"{self.app_name} analytics failed: {error}"
            )
            self.error_occurred.emit(str(runtime_error), traceback.format_exc())
        finally:
            self.processing_finished.emit()

    def process_frame_result(self, frame: QImage) -> FrameResult | None:
        processor = getattr(self.pipeline, "process_frame_result", None)
        if not callable(processor):
            return None

        result = processor(frame)
        if not isinstance(result, FrameResult):
            raise RuntimeError(
                f"{self.app_name} returned {type(result).__name__}; expected FrameResult."
            )
        return result


class ObjectCountAnalytics:
    def update(self, results):
        if isinstance(results, FrameResult):
            return {"current_objects": len(results.detections)}
        return {"current_objects": len(results)}


class MainWindow(QMainWindow):
    analytics_frame_requested = Signal(QImage)

    def __init__(self, initial_app_key: str | None = None):
        super().__init__()

        self.initial_app_key = initial_app_key
        self.available_apps: list[AppDefinition] = []
        self.app_by_key: dict[str, AppDefinition] = {}
        self.camera_registry = CameraRegistry()
        self.available_camera_sources: list[CameraSourceDefinition] = []
        self.camera_source_by_key: dict[str, CameraSourceDefinition] = {}
        self.selected_app: AppDefinition | None = None
        self.selected_camera_source: str | None = None
        self.active_pipeline = None
        self._analytics_busy = False
        self._analytics_enabled = False
        self._analytics_thread = None
        self._analytics_worker = None

        self.setWindowTitle("Vision Analytics")
        self.setGeometry(100, 100, 862, 875)
        self.initUI()

    def initUI(self):
        ui_file = QFile(resource_path("ui/analytics_demo.ui"))
        if not ui_file.open(QFile.ReadOnly):
            raise RuntimeError(f"Cannot open UI file: {resource_path('ui/analytics_demo.ui')}")

        loader = QUiLoader()
        self.ui = loader.load(ui_file)
        ui_file.close()

        try:
            if self.ui is None:
                raise RuntimeError(loader.errorString())

            self.setCentralWidget(self.ui)
            self.performance_metrics()
            self.setup_right_aligned_header_widgets()
            self.setup_results_widgets()
            self.setup_model_profile_widgets()
            self.setup_camera_feed_widget()
            self.setup_camera_source_selector()
            self.setup_application_selector()
        except RuntimeError as error:
            if hasattr(self, "lbl_liveinference_status") and self.lbl_liveinference_status:
                self.lbl_liveinference_status.setText(str(error))
            QMessageBox.critical(self, "UI error", str(error))

    def performance_metrics(self):
        self.lbl_FPS = self.ui.findChild(QLabel, "lbl_FPS")
        self.lbl_Latency = self.ui.findChild(QLabel, "lbl_Latency")
        self.lbl_Resolution = self.ui.findChild(QLabel, "lbl_Resolution")

        missing_widgets = [
            name
            for name, widget in {
                "lbl_FPS": self.lbl_FPS,
                "lbl_Latency": self.lbl_Latency,
                "lbl_Resolution": self.lbl_Resolution,
            }.items()
            if widget is None
        ]

        if missing_widgets:
            raise RuntimeError(f"Missing UI widgets: {', '.join(missing_widgets)}")

        self.lbl_Resolution.setMinimumWidth(64)
        self.lbl_Resolution.setMaximumWidth(80)
        self.reset_performance_metrics()

    def setup_camera_feed_widget(self):
        self.frm_videofeed = self.ui.findChild(QFrame, "frm_videofeed")

        missing_widgets = [
            name
            for name, widget in {
                "frm_videofeed": self.frm_videofeed,
            }.items()
            if widget is None
        ]

        if missing_widgets:
            raise RuntimeError(f"Missing UI widgets: {', '.join(missing_widgets)}")

        self.camera_viewer = attach_camera_viewer(
            self.frm_videofeed,
            auto_start=False,
        )
        self.camera_viewer.metrics_changed.connect(self.update_performance_metrics)
        self.camera_viewer.frame_ready.connect(self.queue_analytics_frame)
        self.camera_viewer.error_occurred.connect(self.handle_camera_error)

    def start_selected_camera_source(self):
        if not hasattr(self, "camera_viewer"):
            return

        source_key = self.current_camera_source_key()
        if not source_key:
            self.stop_camera()
            self.set_live_inference_status("Select Camera", is_live=False)
            return

        self.reset_performance_metrics()
        try:
            camera_config = self.camera_registry.get(source_key)
            camera = CameraFactory.create(camera_config)
        except RuntimeError as error:
            self.show_runtime_error("Camera Config Error", str(error))
            return

        try:
            self.camera_viewer.start(camera.to_camera_config())
            self.selected_camera_source = source_key
        except RuntimeError as error:
            self.set_live_inference_status(str(error), is_live=False)

    def start_usb_camera(self, usb_device: str = "/dev/video0"):
        if hasattr(self, "camera_viewer"):
            self.reset_performance_metrics()
            self.camera_viewer.start_usb(usb_device=usb_device)

    def start_rtsp_camera(self, rtsp_uri: str):
        if hasattr(self, "camera_viewer"):
            self.reset_performance_metrics()
            self.camera_viewer.start_rtsp(rtsp_uri)

    def stop_camera(self):
        if hasattr(self, "camera_viewer"):
            self.clear_video_overlays()
            self.camera_viewer.stop()
        self.reset_performance_metrics()

    def update_performance_metrics(self, metrics: CameraMetrics):
        fps_text = "--" if metrics.fps <= 0 else f"{metrics.fps:.1f}"
        latency_text = (
            "--"
            if metrics.latency_ms is None
            else f"{round(metrics.latency_ms):.0f}ms"
        )

        self.lbl_FPS.setText(fps_text)
        self.lbl_Latency.setText(latency_text)
        self.lbl_Resolution.setText(f"{metrics.width}x{metrics.height}")

    def reset_performance_metrics(self):
        self.lbl_FPS.setText("--")
        self.lbl_Latency.setText("--")
        self.lbl_Resolution.setText("--")

    def setup_results_widgets(self):
        self.lbl_Objects = self.ui.findChild(QLabel, "lbl_Objects")

        if self.lbl_Objects is None:
            raise RuntimeError("Missing UI widgets: lbl_Objects")

        self.lbl_Objects.setText("0")

    def setup_model_profile_widgets(self):
        self.model_profile_labels = {
            "base_model": self.find_first_child(
                QLabel,
                "lbl_basemodel",
                "lbl_base_model",
                "project_title_11",
            ),
            "framework": self.find_first_child(
                QLabel,
                "lbl_framework",
                "project_title_13",
            ),
            "device": self.find_first_child(
                QLabel,
                "lbl_device",
                "project_title_15",
            ),
            "model_input_size": self.find_first_child(
                QLabel,
                "lbl_model_input_size",
                "project_title_17",
            ),
            "confidence_threshold": self.find_first_child(
                QLabel,
                "lbl_confidence_threshold",
                "lbl_confidence_threshold_value",
                "project_title_20",
            ),
        }
        self.hslider_confidence_threshold = self.find_first_child(
            QSlider,
            "hslider_confidence_threshold",
            "horizontalSlider",
        )

        if self.hslider_confidence_threshold is not None:
            self.hslider_confidence_threshold.setRange(0, 100)
            self.hslider_confidence_threshold.setSingleStep(1)
            self.hslider_confidence_threshold.setPageStep(5)
            self.hslider_confidence_threshold.valueChanged.connect(
                self.handle_confidence_threshold_changed
            )

        self.reset_model_profile()

    def find_first_child(self, widget_type, *object_names):
        for object_name in object_names:
            widget = self.ui.findChild(widget_type, object_name)
            if widget is not None:
                return widget
        return None

    def reset_model_profile(self):
        for label in self.model_profile_labels.values():
            if label is not None:
                label.setText("--")
                label.setToolTip("")

        if self.hslider_confidence_threshold is not None:
            self.hslider_confidence_threshold.blockSignals(True)
            self.hslider_confidence_threshold.setValue(0)
            self.hslider_confidence_threshold.setEnabled(False)
            self.hslider_confidence_threshold.blockSignals(False)

    def update_model_profile(self, pipeline):
        profile = self.pipeline_model_profile(pipeline)
        if not profile:
            self.reset_model_profile()
            return

        self.set_model_profile_label("base_model", profile.get("base_model"))
        self.set_model_profile_label("framework", profile.get("framework"))
        self.set_model_profile_label("device", profile.get("device"))
        self.set_model_profile_label(
            "model_input_size",
            profile.get("model_input_size"),
        )

        confidence_threshold = self.profile_confidence_threshold(profile)
        if confidence_threshold is None:
            self.set_model_profile_label("confidence_threshold", "--")
            if self.hslider_confidence_threshold is not None:
                self.hslider_confidence_threshold.setEnabled(False)
            return

        self.set_model_profile_label(
            "confidence_threshold",
            f"{confidence_threshold:.2f}",
        )
        if self.hslider_confidence_threshold is not None:
            self.hslider_confidence_threshold.blockSignals(True)
            self.hslider_confidence_threshold.setValue(
                round(confidence_threshold * 100)
            )
            self.hslider_confidence_threshold.setEnabled(True)
            self.hslider_confidence_threshold.blockSignals(False)

    def pipeline_model_profile(self, pipeline) -> dict:
        profile_getter = getattr(pipeline, "model_profile", None)
        if not callable(profile_getter):
            return {}

        try:
            profile = profile_getter()
        except Exception:
            return {}

        return profile if isinstance(profile, dict) else {}

    def profile_confidence_threshold(self, profile: dict) -> float | None:
        value = profile.get("confidence_threshold")
        if value is None:
            return None

        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return None

    def set_model_profile_label(self, key: str, value):
        label = self.model_profile_labels.get(key)
        if label is None:
            return

        text = str(value) if value not in (None, "") else "--"
        label.setText(text)
        label.setToolTip("" if text == "--" else text)

    @Slot(int)
    def handle_confidence_threshold_changed(self, value: int):
        confidence_threshold = max(0.0, min(1.0, int(value) / 100.0))
        self.set_model_profile_label(
            "confidence_threshold",
            f"{confidence_threshold:.2f}",
        )

        pipeline = self.active_pipeline
        if pipeline is None:
            return

        setter = getattr(pipeline, "set_confidence_threshold", None)
        if callable(setter):
            setter(confidence_threshold)

    def setup_camera_source_selector(self):
        self.available_camera_sources = self.camera_registry.list_sources()
        self.camera_source_by_key = {
            source.key: source for source in self.available_camera_sources
        }

        self.cbo_camera_source.blockSignals(True)
        self.cbo_camera_source.clear()
        self.cbo_camera_source.addItem("Select Camera", "")
        for source in self.available_camera_sources:
            self.cbo_camera_source.addItem(source.label, source.key)
        self.cbo_camera_source.setEnabled(bool(self.available_camera_sources))
        self.cbo_camera_source.blockSignals(False)

        self.cbo_camera_source.currentIndexChanged.connect(
            self.handle_camera_source_changed
        )

    def setup_application_selector(self):
        self.available_apps = discover_app_configs()
        self.app_by_key = {app.key: app for app in self.available_apps}

        self.cbo_selected_test.blockSignals(True)
        self.cbo_selected_test.clear()
        self.cbo_selected_test.addItem("Select Application", "")
        for app in self.available_apps:
            label = app.name if app.enabled else f"{app.name} (Not Implemented)"
            self.cbo_selected_test.addItem(label, app.key)
            if not app.enabled:
                item = self.cbo_selected_test.model().item(
                    self.cbo_selected_test.count() - 1
                )
                if item is not None:
                    item.setEnabled(False)
        self.cbo_selected_test.setEnabled(bool(self.available_apps))
        self.cbo_selected_test.blockSignals(False)

        self.cbo_selected_test.currentIndexChanged.connect(
            self.handle_selected_app_changed
        )

        if not self.available_apps:
            self.set_live_inference_status("No Apps Found", is_live=False)
            return

        if self.initial_app_key in self.app_by_key:
            index = self.cbo_selected_test.findData(self.initial_app_key)
            self.cbo_selected_test.blockSignals(True)
            self.cbo_selected_test.setCurrentIndex(index)
            self.cbo_selected_test.blockSignals(False)
            self.handle_selected_app_changed(index)
            return

        self.set_live_inference_status("Select App", is_live=False)

    @Slot(int)
    def handle_selected_app_changed(self, index: int):
        app_key = self.cbo_selected_test.itemData(index)
        self.stop_active_analytics()
        self.stop_camera()
        self.selected_app = None
        self.active_pipeline = None
        self.lbl_Objects.setText("0")
        self.clear_video_overlays()
        self.reset_model_profile()

        if not app_key:
            self.set_live_inference_status("Select App", is_live=False)
            return

        app = self.app_by_key.get(str(app_key))
        if app is None:
            self.set_live_inference_status("App Missing", is_live=False)
            return
        if not app.enabled:
            self.set_live_inference_status("App Disabled", is_live=False)
            return

        self.selected_app = app

        try:
            pipeline, analytics = self.create_app_runtime(app)
        except Exception as error:
            self.lbl_Objects.setText("--")
            message = self.runtime_error_message(
                error,
                f"{app.name} failed to start",
            )
            self.show_runtime_error(
                f"{app.name} Error",
                message,
                traceback.format_exc(),
            )
            return

        self.active_pipeline = pipeline
        self.update_model_profile(pipeline)

        try:
            self.select_app_default_camera(app)
        except RuntimeError as error:
            self.lbl_Objects.setText("--")
            self.show_runtime_error(
                "Camera Source Error",
                str(error),
                traceback.format_exc(),
            )
            return

        self.start_analytics_worker(pipeline, analytics)
        self.start_selected_camera_source()

    @Slot(int)
    def handle_camera_source_changed(self, _index: int):
        self.selected_camera_source = self.current_camera_source_key()
        if self.selected_app is None or not self._analytics_enabled:
            return

        self.stop_camera()
        self.start_selected_camera_source()

    def select_app_default_camera(self, app: AppDefinition):
        source_name = app.config.get("source", {}).get("name")
        if not source_name:
            return
        if source_name not in self.camera_source_by_key:
            raise RuntimeError(
                f"{app.name} uses camera source '{source_name}', but it is not "
                "defined in configs/cameras.yaml."
            )

        index = self.cbo_camera_source.findData(source_name)
        if index < 0:
            return

        self.cbo_camera_source.blockSignals(True)
        self.cbo_camera_source.setCurrentIndex(index)
        self.cbo_camera_source.blockSignals(False)
        self.selected_camera_source = source_name

    def current_camera_source_key(self) -> str:
        value = self.cbo_camera_source.currentData()
        return "" if value is None else str(value)

    def create_app_runtime(self, app: AppDefinition):
        try:
            pipeline_module = importlib.import_module(
                f"apps.{app.pipeline_type}.pipeline"
            )
        except RuntimeError:
            raise
        except Exception as error:
            raise RuntimeError(
                f"Failed to import {app.name} pipeline "
                f"'{app.pipeline_type}': {error}"
            ) from error

        create_pipeline = getattr(pipeline_module, "create_pipeline_from_config", None)
        if create_pipeline is None:
            raise RuntimeError(f"{app.name} has no pipeline factory.")

        try:
            pipeline = create_pipeline(app.config_path)
            analytics = self.create_app_analytics(app)
        except RuntimeError:
            raise
        except Exception as error:
            raise RuntimeError(
                f"Failed to initialize {app.name}: {error}"
            ) from error

        return pipeline, analytics

    def create_app_analytics(self, app: AppDefinition):
        for module_key in dict.fromkeys((app.key, app.pipeline_type)):
            analytics_module = self.import_optional_analytics_module(module_key)
            if analytics_module is None:
                continue

            for class_name in (
                "Analytics",
                f"{self.class_name_from_key(app.key)}Analytics",
                f"{self.class_name_from_key(module_key)}Analytics",
                "PeopleAnalytics",
            ):
                analytics_class = getattr(analytics_module, class_name, None)
                if analytics_class is not None:
                    return analytics_class()

        return ObjectCountAnalytics()

    def import_optional_analytics_module(self, module_key: str):
        module_name = f"apps.{module_key}.analytics"
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name in {f"apps.{module_key}", module_name}:
                return None
            raise

    def class_name_from_key(self, key: str) -> str:
        return "".join(part.capitalize() for part in key.split("_"))

    def start_analytics_worker(self, pipeline, analytics):
        self._analytics_worker = AnalyticsWorker(
            pipeline,
            analytics,
            self.selected_app.name if self.selected_app else "Selected app",
        )
        self._analytics_thread = QThread(self)
        self._analytics_worker.moveToThread(self._analytics_thread)

        self.analytics_frame_requested.connect(self._analytics_worker.process_frame)
        self._analytics_thread.finished.connect(self._analytics_worker.deleteLater)
        self._analytics_worker.results_ready.connect(
            self.update_analytics_results
        )
        self._analytics_worker.error_occurred.connect(
            self.handle_analytics_error
        )
        self._analytics_worker.processing_finished.connect(
            self.mark_analytics_idle
        )
        self._analytics_thread.start()

        self._analytics_enabled = True
        self.set_live_inference_status("Live Inference", is_live=True)

    @Slot(QImage)
    def queue_analytics_frame(self, frame: QImage):
        if not self._analytics_enabled or self._analytics_busy:
            return

        self._analytics_busy = True
        self.analytics_frame_requested.emit(frame.copy())

    @Slot(object, object, object)
    def update_analytics_results(self, results, summary, source_size=None):
        current_count = summary.get("current_objects")
        if current_count is None:
            current_count = summary.get("current_people", len(results or []))
        self.lbl_Objects.setText(str(int(current_count)))
        self.update_video_overlays(results, source_size)

    @Slot(str, str)
    def handle_analytics_error(self, message: str, details: str = ""):
        self._analytics_enabled = False
        self.lbl_Objects.setText("--")
        self.clear_video_overlays()
        self.show_runtime_error("Analytics Error", message, details)

    @Slot(str)
    def handle_camera_error(self, message: str):
        self.reset_performance_metrics()
        self.clear_video_overlays()
        self.show_runtime_error("Camera Error", message)

    @Slot()
    def mark_analytics_idle(self):
        self._analytics_busy = False

    def set_live_inference_status(self, text: str, is_live: bool):
        if hasattr(self, "lbl_liveinference_status") and self.lbl_liveinference_status:
            display_text = text if is_live or len(text) <= 18 else "Inference Offline"
            self.lbl_liveinference_status.setText(display_text)
            self.lbl_liveinference_status.setToolTip("" if is_live else text)

        if not hasattr(self, "lbl_dotlive_indication") or not self.lbl_dotlive_indication:
            return

        dot_color = "#12B76A" if is_live else "#F04438"
        self.lbl_dotlive_indication.setStyleSheet(
            f"border-radius: 5px; background-color: {dot_color};"
        )

    def show_runtime_error(
        self,
        title: str,
        message: str,
        details: str = "",
    ):
        self.set_live_inference_status(message, is_live=False)

        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Critical)
        dialog.setWindowTitle(title)
        dialog.setText(message)
        if details:
            dialog.setDetailedText(details)
        dialog.exec()

    def runtime_error_message(self, error: Exception, fallback: str) -> str:
        if isinstance(error, RuntimeError) and str(error):
            return str(error)
        if str(error):
            return f"{fallback}: {error}"
        return fallback

    def update_video_overlays(self, results, source_size=None):
        if not hasattr(self, "camera_viewer"):
            return
        if not hasattr(self.camera_viewer, "set_overlays"):
            return
        self.camera_viewer.set_overlays(results, source_size)

    def clear_video_overlays(self):
        if not hasattr(self, "camera_viewer"):
            return
        if not hasattr(self.camera_viewer, "clear_overlays"):
            return
        self.camera_viewer.clear_overlays()

    def setup_right_aligned_header_widgets(self):
        self.header_frame = self.ui.findChild(QFrame, "frame_3")
        self.cbo_selected_test = self.ui.findChild(QComboBox, "cbo_selected_test")
        self.lbl_dotlive_indication = self.ui.findChild(QLabel, "lbl_dotlive_indication")
        self.lbl_liveinference_status = self.ui.findChild(QLabel, "lbl_liveinference_status")

        missing_widgets = [
            name
            for name, widget in {
                "frame_3": self.header_frame,
                "cbo_selected_test": self.cbo_selected_test,
                "lbl_dotlive_indication": self.lbl_dotlive_indication,
                "lbl_liveinference_status": self.lbl_liveinference_status,
            }.items()
            if widget is None
        ]

        if missing_widgets:
            raise RuntimeError(f"Missing UI widgets: {', '.join(missing_widgets)}")

        self.live_status_container = self.lbl_liveinference_status.parentWidget()
        self.cbo_camera_source = QComboBox(self.header_frame)
        self.cbo_camera_source.setObjectName("cbo_camera_source")
        self.cbo_camera_source.setGeometry(self.cbo_selected_test.geometry())
        self.cbo_camera_source.setStyleSheet(self.cbo_selected_test.styleSheet())
        self.cbo_camera_source.setFixedSize(170, self.cbo_selected_test.height())
        self.cbo_selected_test.setFixedWidth(170)
        self._header_right_margin = 5
        self._header_widget_gap = 8
        self._cbo_selected_test_y = self.cbo_selected_test.y()
        self._cbo_camera_source_y = self.cbo_selected_test.y()
        self._live_status_container_y = self.live_status_container.y()

        self.header_frame.installEventFilter(self)
        self.position_header_right_widgets()

    def eventFilter(self, watched, event):
        if watched is self.header_frame and event.type() == QEvent.Type.Resize:
            self.position_header_right_widgets()
        return super().eventFilter(watched, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.position_header_right_widgets()

    def closeEvent(self, event):
        self.stop_camera()
        self.stop_active_analytics()
        super().closeEvent(event)

    def stop_active_analytics(self):
        self._analytics_enabled = False
        self._analytics_busy = False

        if self._analytics_thread is None:
            return

        if self._analytics_worker is not None:
            try:
                self.analytics_frame_requested.disconnect(
                    self._analytics_worker.process_frame
                )
            except (RuntimeError, TypeError):
                pass

        self._analytics_thread.quit()
        self._analytics_thread.wait(2000)
        self._analytics_thread = None
        self._analytics_worker = None

    def position_header_right_widgets(self):
        if not hasattr(self, "header_frame"):
            return

        status_x = (
            self.header_frame.width()
            - self._header_right_margin
            - self.live_status_container.width()
        )
        camera_x = status_x - self._header_widget_gap - self.cbo_camera_source.width()
        app_x = camera_x - self._header_widget_gap - self.cbo_selected_test.width()

        self.live_status_container.move(
            max(self._header_right_margin, status_x),
            self._live_status_container_y,
        )
        self.cbo_camera_source.move(
            max(self._header_right_margin, camera_x),
            self._cbo_camera_source_y,
        )
        self.cbo_selected_test.move(
            max(self._header_right_margin, app_x),
            self._cbo_selected_test_y,
        )


# ── Standalone entry point ─────────────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
