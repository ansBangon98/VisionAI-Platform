from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from ui.analytics_demo import MainWindow


class PeopleAnalyticsController:
    """Connect the people analytics pipeline to the Qt application window."""

    def __init__(self):
        self.app = QApplication.instance() or QApplication(sys.argv)
        self.window = self._create_window()

    def run(self) -> int:
        self.window.show()
        return self.app.exec()

    def _create_window(self):
        return MainWindow(initial_app_key="people_analytics")
