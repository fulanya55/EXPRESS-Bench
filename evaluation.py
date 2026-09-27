import numpy as np
import re


def _parse_mark(value):
    """Parse either `3` or the rubric-style `3/5` returned by the judge."""
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    if match is None:
        raise ValueError(f"Could not parse score component: {value!r}")
    return float(match.group(0))


def _parse_eac(value):
    """Parse the judge's ``grounding, accuracy`` response.

    A small number of gateway responses can be malformed (for example
    ``1/2`` with the accuracy component missing).  Such a row remains in the
    per-question report but cannot contribute to aggregate metrics.
    """
    pieces = value.replace("Your mark:", "").strip().split(",", 1)
    if len(pieces) != 2:
        return None
    try:
        return _parse_mark(pieces[0]), int(_parse_mark(pieces[1]))
    except (TypeError, ValueError):
        return None


def score(results):
    C, C_star, p_path, l_path, d_T = [], [], [], [], []
    for result in results:
        if result["path_len"] != float("inf"):
            EAC = result["EAC"].replace("Your mark:", "").strip()
            parsed_eac = _parse_eac(EAC)
            if parsed_eac is None:
                continue
            grd, acc = parsed_eac
            C.append(grd*acc)
            C_star.append(acc)
            p_path.append(result["path_len"])
            l_path.append(result["geodesic_distance"])
        if result["goal_dis"] != float("inf"):
            d_T.append(result["goal_dis"])

    weight_path = l_path / np.maximum(p_path, l_path)
    C_avg = np.mean(100.0 * (np.clip(C, 0, 5) / 5))
    C_star_avg = np.mean(100.0 * (np.clip(C_star, 0, 5) / 5))
    E_path = np.mean(100.0 * (np.clip(C, 0, 5) / 5) * weight_path)
    d_T_avg = np.mean(d_T)

    return C_avg, C_star_avg, E_path, d_T_avg
