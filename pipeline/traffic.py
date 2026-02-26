"""
Standalone traffic analysis logic.
No dependencies on VideoProcessor, Redis, or OpenCV.
"""


def analyze_traffic(detections: list) -> dict:
    """
    Derive traffic density and vehicle counts from a list of detections.
    Density is calculated from vehicles only — persons are tracked separately
    but do not contribute to traffic density, as they are not vehicles.
    """
    counts = {
        "cars": 0,
        "motorcycles": 0,
        "trucks": 0,
        "buses": 0,
        "bicycles": 0,
        "persons": 0,   # tracked for display but excluded from density
        "total_vehicles": 0,
        "total_detections": len(detections),
    }

    for det in detections:
        class_name = det["class_name"].lower()
        if class_name == "car":
            counts["cars"] += 1
        elif class_name in ("motorcycle", "motorbike"):
            counts["motorcycles"] += 1
        elif class_name == "truck":
            counts["trucks"] += 1
        elif class_name == "bus":
            counts["buses"] += 1
        elif class_name == "bicycle":
            counts["bicycles"] += 1
        elif class_name == "person":
            counts["persons"] += 1

    # Density based on vehicles only
    counts["total_vehicles"] = (
        counts["cars"] +
        counts["motorcycles"] +
        counts["trucks"] +
        counts["buses"] +
        counts["bicycles"]
    )

    max_expected_vehicles = 20
    density = min(100.0, (counts["total_vehicles"] / max_expected_vehicles) * 100)

    if density < 30:
        status, status_color, status_icon = "LIGHT", "green", "🟢"
    elif density < 70:
        status, status_color, status_icon = "MODERATE", "orange", "🟡"
    else:
        status, status_color, status_icon = "HEAVY", "red", "🔴"

    return {
        "counts": counts,
        "density": round(density, 1),
        "status": status,
        "status_color": status_color,
        "status_icon": status_icon,
    }