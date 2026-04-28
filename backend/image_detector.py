def _jsonable(obj):
    """Convert DeepFace/numpy output so FastAPI can return JSON."""
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore

    if np is not None:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return _jsonable(vars(obj))
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def analyze_image(image_path):
    if not image_path:
        raise ValueError("Image path is required.")

    try:
        from deepface import DeepFace

        result = DeepFace.analyze(image_path, actions=['emotion', 'age', 'gender'], enforce_detection=True)
        return {
            "real": True,
            "analysis": _jsonable(result),
        }
    except ValueError as exc:
        raise ValueError(f"Image validation failed: {exc}") from exc
    except RuntimeError as exc:
        raise RuntimeError(f"Image analysis failed: {exc}") from exc

