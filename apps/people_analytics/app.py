from __future__ import annotations

from apps.people_analytics.controller import PeopleAnalyticsController


def main() -> int:
    controller = PeopleAnalyticsController()
    return controller.run()


if __name__ == "__main__":
    raise SystemExit(main())
