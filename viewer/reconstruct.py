"""Subprocess boundary around the published DXA-to-3D checkpoint."""
import argparse
import json
import os
from pathlib import Path


def validate_points(points):
    import math
    if not isinstance(points, list) or not 2 <= len(points) <= 10000:
        raise ValueError("Expected Nx3 points")
    if any(not isinstance(p, list) or len(p) != 3 or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in p) for p in points):
        raise ValueError("Invalid 3D coordinates")
    return points


def curves_to_points(curves):
    """Combine predicted coronal and sagittal centre lines into a 3D curve."""
    import numpy as np
    values = np.asarray(curves, dtype=float)
    if values.shape != (209, 6) or not np.isfinite(values).all():
        raise ValueError("Expected 209x6 finite curve predictions")
    vertical = np.linspace(0.0, 1.0, len(values))
    return np.column_stack((values[:, 1], vertical, values[:, 4])).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkpoint = Path(os.environ["DXA3D_CHECKPOINT"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("DXA-to-3D checkpoint missing")
    from viewer.dxa3d_published import predict_curves
    points = validate_points(curves_to_points(predict_curves(args.input, checkpoint)))
    Path(args.output).write_text(json.dumps({"points": points, "units": "normalized_model_coordinates",
                                           "model": "DXA-to-3D", "experimental": True}, allow_nan=False))


if __name__ == "__main__":
    main()
