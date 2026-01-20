"""
Shared camera configurations for rendering.
These are used across different rendering scripts to ensure consistency.
"""

from mathutils import Vector


FRONTAL_VIEWS = [
    {"dir": Vector((0, 1, 0)), "weight": 15},         # Dead center front
    {"dir": Vector((0.15, 1, 0)), "weight": 12},      # Slight right
    {"dir": Vector((-0.15, 1, 0)), "weight": 12},     # Slight left
    {"dir": Vector((0.3, 0.95, 0)), "weight": 10},    # Moderate right
    {"dir": Vector((-0.3, 0.95, 0)), "weight": 10},   # Moderate left
    {"dir": Vector((0.5, 0.85, 0)), "weight": 8},     # 3/4 view right
    {"dir": Vector((-0.5, 0.85, 0)), "weight": 8},    # 3/4 view left
    {"dir": Vector((0.7, 0.7, 0)), "weight": 5},      # Wide angle right
    {"dir": Vector((-0.7, 0.7, 0)), "weight": 5},     # Wide angle left
]

SIDE_VIEWS = [
    {"dir": Vector((0.9, 0.4, 0)), "weight": 3},
    {"dir": Vector((-0.9, 0.4, 0)), "weight": 3},
]

CAMERA_HEIGHTS = [
    {"z": 0.0, "name": "eye_level", "weight": 50},
    {"z": 0.12, "name": "slightly_up", "weight": 30},
    {"z": -0.08, "name": "slightly_down", "weight": 20},
    {"z": 0.25, "name": "low_angle", "weight": 10},
]

PHONE_CAMERA_CONFIGS = [
    {
        "name": "Normal",
        "lens": 28,
        "distance_factor": 1.4,
        "weight": 70,
        "tilt_range": (-2, 2),
        "position_jitter": 0.02
    },
    {
        "name": "Standard_Close",
        "lens": 28,
        "distance_factor": 1.1,
        "weight": 40,
        "tilt_range": (-3, 3),
        "position_jitter": 0.025
    },
    {
        "name": "Wide",
        "lens": 24,
        "distance_factor": 1.0,
        "weight": 30,
        "tilt_range": (-3, 3),
        "position_jitter": 0.03
    },
    {
        "name": "Telephoto",
        "lens": 52,
        "distance_factor": 2.2,
        "weight": 20,
        "tilt_range": (-1, 1),
        "position_jitter": 0.015
    },
]


def get_all_views():
    """Get combined list of all view directions"""
    return FRONTAL_VIEWS + SIDE_VIEWS
