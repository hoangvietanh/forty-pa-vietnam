"""Forest Condition Index (FCI): one score per PA and year, and its change between the first and last year.

FCI = 100 * sum(weight of class * class area) / area of the PA. A PA is called declining or recovering only when the
change is at least MIN_DELTA points and the 9-year series has a monotonic trend (Mann-Kendall p < MAX_P); otherwise it
is stable / unclear."""
from itertools import combinations
from math import erf, sqrt

import numpy as np
import pandas as pd

# How natural a class is, from 1 (primary mature forest) to 0 (other land).
WEIGHTS = {1: 1.0, 7: 0.8, 2: 0.6, 3: 0.3, 4: 0.2, 5: 0.1, 6: 0.0}
MIN_DELTA = 1.0
MAX_P = 0.05
DECLINE, RECOVERY, STABLE = "Suy giảm", "Phục hồi", "Ổn định / không rõ"


def mann_kendall_p(y):
    """Two-sided p-value of the Mann-Kendall trend test (normal approximation, no tie correction)."""
    n = len(y)
    s = sum(np.sign(y[j] - y[i]) for i, j in combinations(range(n), 2))
    if s == 0:
        return 1.0
    z = (s - np.sign(s)) / sqrt(n * (n - 1) * (2 * n + 5) / 18)
    return 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))


def theil_sen(y, x):
    """Median of the pairwise slopes."""
    return float(np.median([(y[j] - y[i]) / (x[j] - x[i]) for i, j in combinations(range(len(x)), 2)]))


def fci_series(areas, years):
    """FCI per PA (rows) and year (columns) from the long area table of gee.compute_areas."""
    a = areas[areas.year.isin(years)].assign(w=lambda d: d.value.map(WEIGHTS) * d.area_ha)
    g = a.groupby(["SITE_ID", "year"])[["w", "area_ha"]].sum()
    return (100 * g.w / g.area_ha).unstack("year").reindex(columns=list(years))


def fci_index(areas, years):
    """One row per PA: FCI of the first and last year, their difference, the Theil-Sen trend over the whole period,
    the Mann-Kendall p-value and the verdict."""
    y0, y1 = years[0], years[-1]
    s = fci_series(areas, years).dropna()
    x = np.array(years, dtype=float)
    rows = []
    for site_id, r in s.iterrows():
        v = r.to_numpy(dtype=float)
        delta, p = v[-1] - v[0], mann_kendall_p(v)
        verdict = STABLE if abs(delta) < MIN_DELTA or p >= MAX_P else DECLINE if delta < 0 else RECOVERY
        rows.append({"SITE_ID": site_id, f"fci_{y0}": v[0], f"fci_{y1}": v[-1], "dfci": delta,
                     "trend": theil_sen(v, x) * (y1 - y0), "mk_p": p, "verdict": verdict})
    return pd.DataFrame(rows)
