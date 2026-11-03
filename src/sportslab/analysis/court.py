"""NBA half-court geometry in ShotChartDetail units (tenths of a foot, hoop at the origin).

``x`` runs from -250 to 250 across the baseline, ``y`` from -47.5 (baseline) to 422.5 (half court).
"""

import numpy as np

BASELINE_Y = -47.5
HALF_COURT_Y = 422.5
RIM_RADIUS = 7.5
THREE_POINT_RADIUS = 237.5
CORNER_THREE_X = 220.0


def _arc(radius: float, start: float, end: float, cx: float = 0.0, cy: float = 0.0, n: int = 60):
    theta = np.linspace(start, end, n)
    return cx + radius * np.cos(theta), cy + radius * np.sin(theta)


def court_lines() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Named polylines (x, y) that draw a half court."""
    corner_y = float(np.sqrt(THREE_POINT_RADIUS**2 - CORNER_THREE_X**2))
    corner_angle = np.arctan2(corner_y, CORNER_THREE_X)
    arc_x, arc_y = _arc(THREE_POINT_RADIUS, corner_angle, np.pi - corner_angle)

    def box(half_width: float, top: float):
        return np.array([-half_width, -half_width, half_width, half_width]), np.array(
            [BASELINE_Y, top, top, BASELINE_Y]
        )

    lines = {
        "boundary": (
            np.array([-250, -250, 250, 250, -250]),
            np.array([BASELINE_Y, HALF_COURT_Y, HALF_COURT_Y, BASELINE_Y, BASELINE_Y]),
        ),
        "paint": box(80, 142.5),
        "lane": box(60, 142.5),
        "backboard": (np.array([-30.0, 30.0]), np.array([-7.5, -7.5])),
        "rim": _arc(RIM_RADIUS, 0, 2 * np.pi),
        "restricted_area": _arc(40, 0, np.pi),
        "free_throw_top": _arc(60, 0, np.pi, cy=142.5),
        "free_throw_bottom": _arc(60, np.pi, 2 * np.pi, cy=142.5),
        "three_point_arc": (arc_x, arc_y),
        "three_point_left": (np.array([-CORNER_THREE_X] * 2), np.array([BASELINE_Y, corner_y])),
        "three_point_right": (np.array([CORNER_THREE_X] * 2), np.array([BASELINE_Y, corner_y])),
        "center_circle": _arc(60, np.pi, 2 * np.pi, cy=HALF_COURT_Y),
    }
    return {
        name: (np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        for name, (x, y) in lines.items()
    }
