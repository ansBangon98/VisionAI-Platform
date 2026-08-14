import sys
from datetime import datetime

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QFrame,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QSizePolicy,
)


# ============================================================
# BASE CARD
# ============================================================

class DashboardCard(QFrame):
    """
    Base reusable card used by all dashboard widgets.
    """

    def __init__(self, title: str, parent=None):
        super().__init__(parent)

        self.setObjectName("dashboardCard")

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(15, 12, 15, 15)
        self.main_layout.setSpacing(10)

        # Title
        self.title_label = QLabel(title.upper())
        self.title_label.setObjectName("cardTitle")

        self.main_layout.addWidget(self.title_label)

        # Divider
        divider = QFrame()
        divider.setObjectName("divider")
        divider.setFixedHeight(1)

        self.main_layout.addWidget(divider)


# ============================================================
# 1. ANALYTICS SUMMARY
# ============================================================

class AnalyticsSummaryWidget(DashboardCard):

    def __init__(self, parent=None):
        super().__init__("Analytics Summary", parent)

        self.person_count = 0
        self.entries = 0
        self.exits = 0
        self.avg_dwell_time = 0

        # Person Count
        (
            self.person_count_value,
            self.person_count_progress
        ) = self._create_progress_metric(
            "Person count",
            "#6697f5",
        )

        # Entries
        (
            self.entries_value,
            self.entries_progress
        ) = self._create_progress_metric(
            "Entries (last 5 min)",
            "#69c29b",
        )

        # Exits
        (
            self.exits_value,
            self.exits_progress
        ) = self._create_progress_metric(
            "Exits (last 5 min)",
            "#edb34e",
        )

        # Average dwell
        dwell_layout = QHBoxLayout()

        dwell_label = QLabel("Avg. dwell time")
        dwell_label.setObjectName("metricLabel")

        self.dwell_value = QLabel("0s")
        self.dwell_value.setObjectName("metricValue")

        dwell_layout.addWidget(dwell_label)
        dwell_layout.addStretch()
        dwell_layout.addWidget(self.dwell_value)

        self.main_layout.addLayout(dwell_layout)

        self.main_layout.addStretch()

    def _create_progress_metric(self, title, color):
        container = QVBoxLayout()
        container.setSpacing(5)

        header = QHBoxLayout()

        label = QLabel(title)
        label.setObjectName("metricLabel")

        value_label = QLabel("0")
        value_label.setObjectName("metricValue")

        header.addWidget(label)
        header.addStretch()
        header.addWidget(value_label)

        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(0)
        progress.setTextVisible(False)
        progress.setFixedHeight(6)

        progress.setStyleSheet(
            f"""
            QProgressBar {{
                background-color: #2b3038;
                border: none;
                border-radius: 3px;
            }}

            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 3px;
            }}
            """
        )

        container.addLayout(header)
        container.addWidget(progress)

        self.main_layout.addLayout(container)

        return value_label, progress

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def update_stats(
        self,
        person_count=None,
        entries=None,
        exits=None,
        avg_dwell_time=None,
    ):

        if person_count is not None:
            self.person_count = person_count
            self.person_count_value.setText(str(person_count))

            # Example visualization scale
            self.person_count_progress.setValue(
                min(int(person_count * 5), 100)
            )

        if entries is not None:
            self.entries = entries
            self.entries_value.setText(str(entries))

            self.entries_progress.setValue(
                min(int(entries * 3), 100)
            )

        if exits is not None:
            self.exits = exits
            self.exits_value.setText(str(exits))

            self.exits_progress.setValue(
                min(int(exits * 3), 100)
            )

        if avg_dwell_time is not None:
            self.avg_dwell_time = avg_dwell_time

            self.dwell_value.setText(
                f"{avg_dwell_time}s"
            )

    def reset(self):
        self.update_stats(
            person_count=0,
            entries=0,
            exits=0,
            avg_dwell_time=0,
        )


# ============================================================
# 2. EVENT HISTORY
# ============================================================

class EventHistoryWidget(DashboardCard):

    def __init__(self, max_items=10, parent=None):
        super().__init__("Event History", parent)

        self.max_items = max_items

        self.events_container = QWidget()
        self.events_layout = QVBoxLayout(
            self.events_container
        )

        self.events_layout.setContentsMargins(0, 0, 0, 0)
        self.events_layout.setSpacing(0)

        self.main_layout.addWidget(
            self.events_container
        )

        self.main_layout.addStretch()

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def add_event(
        self,
        message: str,
        elapsed: str = "0s",
    ):
        """
        Add newest event at the TOP.

        Example:
            add_event(
                "ID 5 entered zone A",
                "-2s"
            )
        """

        event_widget = self._create_event_row(
            elapsed,
            message,
        )

        # Add newest event at TOP
        self.events_layout.insertWidget(
            0,
            event_widget,
        )

        # Remove oldest event
        if self.events_layout.count() > self.max_items:

            last_index = (
                self.events_layout.count() - 1
            )

            item = self.events_layout.takeAt(
                last_index
            )

            widget = item.widget()

            if widget is not None:
                widget.deleteLater()

    def _create_event_row(
        self,
        elapsed,
        message,
    ):

        container = QFrame()
        container.setObjectName("eventRow")

        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 9, 0, 9)
        layout.setSpacing(15)

        time_label = QLabel(elapsed)
        time_label.setObjectName("eventTime")
        time_label.setFixedWidth(47)

        message_label = QLabel(message)
        message_label.setObjectName("eventMessage")

        layout.addWidget(time_label)
        layout.addWidget(message_label)
        layout.addStretch()

        return container

    def clear_events(self):

        while self.events_layout.count():

            item = self.events_layout.takeAt(0)

            widget = item.widget()

            if widget is not None:
                widget.deleteLater()


# ============================================================
# 3. DETECTION RESULTS
# ============================================================

class DetectionResultsWidget(DashboardCard):

    def __init__(
        self,
        max_items=10,
        parent=None,
    ):
        super().__init__(
            "Detection Results — Current Frame",
            parent,
        )

        self.max_items = max_items

        self.table = QTableWidget()

        self.table.setColumnCount(4)

        self.table.setHorizontalHeaderLabels(
            [
                "ID",
                "CLASS",
                "LABEL",
                "CONFIDENCE",
            ]
        )

        self.table.verticalHeader().setVisible(False)

        self.table.setShowGrid(False)

        self.table.setSelectionMode(
            QTableWidget.SelectionMode.NoSelection
        )

        self.table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )

        self.table.setFocusPolicy(
            Qt.FocusPolicy.NoFocus
        )

        self.table.setObjectName(
            "detectionTable"
        )

        header = self.table.horizontalHeader()

        header.setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        self.table.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        self.main_layout.addWidget(self.table)

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def add_detection(
        self,
        track_id,
        class_name,
        label,
        confidence,
    ):
        """
        New detection is inserted on TOP.

        Oldest detection is removed automatically
        when more than max_items exist.
        """

        # Add new row at the TOP
        self.table.insertRow(0)

        id_item = QTableWidgetItem(
            str(track_id)
        )

        class_item = QTableWidgetItem(
            f"■  {class_name}"
        )

        label_item = QTableWidgetItem(
            str(label)
        )

        confidence_item = QTableWidgetItem(
            f"{confidence:.2f}"
        )

        # IDs
        id_item.setTextAlignment(
            Qt.AlignmentFlag.AlignVCenter |
            Qt.AlignmentFlag.AlignLeft
        )

        # Add items
        self.table.setItem(
            0,
            0,
            id_item,
        )

        self.table.setItem(
            0,
            1,
            class_item,
        )

        self.table.setItem(
            0,
            2,
            label_item,
        )

        self.table.setItem(
            0,
            3,
            confidence_item,
        )

        # Remove oldest item
        if self.table.rowCount() > self.max_items:

            last_row = (
                self.table.rowCount() - 1
            )

            self.table.removeRow(
                last_row
            )

    def clear_detections(self):

        self.table.setRowCount(0)

    def set_detections(self, detections):
        """
        Useful when detections represent only
        the CURRENT FRAME.

        detections example:

        [
            {
                "track_id": 1,
                "class_name": "person",
                "label": "person",
                "confidence": 0.85
            },
            ...
        ]
        """

        self.clear_detections()

        for detection in reversed(
            detections[-self.max_items:]
        ):

            self.add_detection(
                track_id=detection["track_id"],
                class_name=detection["class_name"],
                label=detection["label"],
                confidence=detection["confidence"],
            )


# ============================================================
# 4. SESSION STATS
# ============================================================

class SessionStatsWidget(DashboardCard):

    def __init__(self, parent=None):
        super().__init__(
            "Session Stats",
            parent,
        )

        self.uptime_value = self._add_stat(
            "Uptime"
        )

        self.tracked_ids_value = self._add_stat(
            "Total tracked IDs"
        )

        self.peak_occupancy_value = self._add_stat(
            "Peak occupancy"
        )

        self.processing_time_value = self._add_stat(
            "Avg processing time"
        )

        self.main_layout.addStretch()

    def _add_stat(self, name):

        row = QFrame()
        row.setObjectName("statsRow")

        layout = QHBoxLayout(row)

        layout.setContentsMargins(
            0,
            9,
            0,
            9,
        )

        label = QLabel(name)
        label.setObjectName(
            "metricLabel"
        )

        value = QLabel("0")
        value.setObjectName(
            "metricValue"
        )

        layout.addWidget(label)
        layout.addStretch()
        layout.addWidget(value)

        self.main_layout.addWidget(row)

        return value

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def update_stats(
        self,
        uptime=None,
        total_tracked_ids=None,
        peak_occupancy=None,
        avg_processing_time=None,
    ):

        if uptime is not None:
            self.uptime_value.setText(
                str(uptime)
            )

        if total_tracked_ids is not None:
            self.tracked_ids_value.setText(
                str(total_tracked_ids)
            )

        if peak_occupancy is not None:
            self.peak_occupancy_value.setText(
                str(peak_occupancy)
            )

        if avg_processing_time is not None:
            self.processing_time_value.setText(
                f"{avg_processing_time} ms"
            )


# ============================================================
# MAIN DASHBOARD
# ============================================================

class AnalyticsDashboard(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "Vision Analytics Dashboard"
        )

        self.resize(1250, 520)

        main_layout = QGridLayout(self)

        main_layout.setContentsMargins(
            10,
            20,
            10,
            20,
        )

        main_layout.setSpacing(14)

        # ----------------------------------------------------
        # Reusable widgets
        # ----------------------------------------------------

        self.analytics_summary = (
            AnalyticsSummaryWidget()
        )

        self.event_history = (
            EventHistoryWidget(
                max_items=10
            )
        )

        self.detection_results = (
            DetectionResultsWidget(
                max_items=10
            )
        )

        self.session_stats = (
            SessionStatsWidget()
        )

        # ----------------------------------------------------
        # Layout
        # ----------------------------------------------------

        main_layout.addWidget(
            self.analytics_summary,
            0,
            0,
        )

        main_layout.addWidget(
            self.event_history,
            0,
            1,
        )

        main_layout.addWidget(
            self.detection_results,
            1,
            0,
        )

        main_layout.addWidget(
            self.session_stats,
            1,
            1,
        )

        main_layout.setColumnStretch(
            0,
            1,
        )

        main_layout.setColumnStretch(
            1,
            1,
        )

        main_layout.setRowStretch(
            0,
            1,
        )

        main_layout.setRowStretch(
            1,
            1,
        )

        self.apply_styles()

        self.load_example_data()

    # ========================================================
    # DEMO DATA
    # ========================================================

    def load_example_data(self):

        # Analytics
        self.analytics_summary.update_stats(
            person_count=8,
            entries=14,
            exits=11,
            avg_dwell_time=42,
        )

        # ----------------------------------------------------
        # Events
        # ----------------------------------------------------

        self.event_history.add_event(
            "Occupancy peak: 19 people",
            "-44s",
        )

        self.event_history.add_event(
            "ID 1 entered zone A",
            "-31s",
        )

        self.event_history.add_event(
            "ID 2 exited frame (right)",
            "-18s",
        )

        self.event_history.add_event(
            "ID 3 dwell exceeded 60s",
            "-9s",
        )

        self.event_history.add_event(
            "ID 5 entered zone A",
            "-2s",
        )

        # ----------------------------------------------------
        # Detections
        # ----------------------------------------------------

        self.detection_results.add_detection(
            "001",
            "person",
            "person",
            0.55,
        )

        self.detection_results.add_detection(
            "002",
            "person",
            "person",
            0.56,
        )

        self.detection_results.add_detection(
            "003",
            "person",
            "person",
            0.78,
        )

        self.detection_results.add_detection(
            "004",
            "person",
            "person",
            0.60,
        )

        self.detection_results.add_detection(
            "005",
            "person",
            "person",
            0.88,
        )

        self.detection_results.add_detection(
                    "005",
                    "person",
                    "person",
                    0.88,
                )
        self.detection_results.add_detection(
                    "005",
                    "person",
                    "person",
                    0.88,
                )
        self.detection_results.add_detection(
                    "005",
                    "person",
                    "person",
                    0.88,
                )
        self.detection_results.add_detection(
                    "005",
                    "person",
                    "person",
                    0.88,
                )
        self.detection_results.add_detection(
                    "005",
                    "person",
                    "person",
                    0.88,
                )
        self.detection_results.add_detection(
                    "005",
                    "person",
                    "person",
                    0.88,
                )
        self.detection_results.add_detection(
                    "005",
                    "person",
                    "person",
                    0.88,
                )

        # Session
        self.session_stats.update_stats(
            uptime="02:14:08",
            total_tracked_ids=1342,
            peak_occupancy=19,
            avg_processing_time=31,
        )

    # ========================================================
    # STYLE
    # ========================================================

    def apply_styles(self):

        self.setStyleSheet(
            """

            QWidget {
                background-color: #191d22;
                color: #d6dce7;
                font-family: Arial;
                font-size: 13px;
            }


            /* ================================
               CARD
            ================================ */

            QFrame#dashboardCard {

                background-color: #22262c;

                border: 1px solid #2e343d;

                border-radius: 10px;
            }


            QLabel#cardTitle {

                color: #65738b;

                font-size: 12px;

                font-weight: bold;

                letter-spacing: 0.5px;
            }


            QFrame#divider {

                background-color: #30353d;

                border: none;
            }


            /* ================================
               METRICS
            ================================ */

            QLabel#metricLabel {

                color: #8998b2;
            }


            QLabel#metricValue {

                color: #eef2fa;

                font-weight: bold;
            }


            /* ================================
               EVENT HISTORY
            ================================ */

            QFrame#eventRow {

                border: none;

                border-bottom:
                    1px solid #30353d;
            }


            QLabel#eventTime {

                color: #64728a;
            }


            QLabel#eventMessage {

                color: #98a6bd;
            }


            /* ================================
               SESSION
            ================================ */

            QFrame#statsRow {

                border: none;

                border-bottom:
                    1px solid #30353d;
            }


            /* ================================
               DETECTION TABLE
            ================================ */

            QTableWidget#detectionTable {

                background-color: transparent;

                border: none;

                color: #98a6bd;

                gridline-color: transparent;
            }


            QTableWidget#detectionTable::item {

                border-bottom:
                    1px solid #30353d;

                padding: 6px;
            }


            QHeaderView::section {

                background-color: transparent;

                color: #66748b;

                border: none;

                border-bottom:
                    1px solid #30353d;

                padding: 7px;

                font-size: 11px;

                font-weight: bold;
            }


            QTableCornerButton::section {

                background-color: transparent;

                border: none;
            }


            QScrollBar:vertical {

                width: 7px;

                background: transparent;
            }


            QScrollBar::handle:vertical {

                background: #3a414c;

                border-radius: 3px;

                min-height: 20px;
            }


            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {

                height: 0px;
            }

            """
        )


# ============================================================
# APPLICATION
# ============================================================

if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = AnalyticsDashboard()

    window.show()

    sys.exit(app.exec())