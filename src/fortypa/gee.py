"""Earth Engine side: ForTy V2 Beta class maps, protected areas from WDPA, class areas per year, map thumbnails."""
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import requests
from PIL import Image

FORTY = "projects/nature-trace/assets/forest_typology/v2_beta"
WDPA = "WCMC/WDPA/current/polygons"

# Class value = argmax band index + 1 (B0 -> 1 … B6 -> 7). Rows run from natural to planted classes, the order used in
# legends and tables. Forest colors pass the dataviz palette validator (light surface, adjacent pairs in this order).
CLASSES = pd.DataFrame([
    (1, "B0", "Rừng nguyên sinh – trưởng thành", "Primary forest: mature", "#16733A"),
    (7, "B6", "Rừng nguyên sinh – đang phục hồi", "Primary forest: regrowing", "#33A17F"),
    (2, "B1", "Rừng tái sinh tự nhiên", "Naturally regenerating forest", "#A0B82B"),
    (3, "B2", "Rừng trồng (luân kỳ > 40 năm)", "Planted forest", "#2C7FB8"),
    (4, "B3", "Rừng trồng thâm canh (luân kỳ ≤ 40 năm)", "Plantation forest", "#C2408F"),
    (5, "B4", "Cây công nghiệp và nông lâm kết hợp", "Tree crops and agroforestry", "#D08C00"),
    (6, "B5", "Đất khác (không phải rừng)", "Other land (non-forest)", "#E4E4DF"),
], columns=["value", "band", "label", "label_en", "color"])
FOREST = CLASSES[CLASSES.value != 6].reset_index(drop=True)
NATURAL = (1, 7, 2)  # mature primary, regrowing primary, naturally regenerating
PLANTED = (3, 4, 5)  # planted, plantation, tree crops and agroforestry

# Class change between two years, grouped by what changed (N natural, P planted / tree crops, O other land). Four
# colors only: with six, some pair always failed the dataviz validator over all pairs (they all meet on a map). This
# order passes it over all pairs (CVD 7.5 for 3↔2, so the legend carries labels and areas as secondary encoding).
# The per-transition areas (from class -> to class) stay available in the transitions table.
CHANGES = pd.DataFrame([
    (0, "Không đổi lớp", "#E4E4DF"),
    (1, "Đổi lớp trong cùng nhóm (tự nhiên ↔ tự nhiên, trồng ↔ trồng)", "#D9A400"),
    (2, "Tăng rừng / phục hồi (đất khác → rừng, rừng trồng → tự nhiên)", "#1E6FD9"),
    (3, "Rừng / cây → đất khác", "#B83FB0"),
    (4, "Rừng tự nhiên → rừng trồng / cây công nghiệp", "#D62828"),
], columns=["code", "label", "color"])


def _group(c):
    return "N" if c in NATURAL else "P" if c in PLANTED else "O"


def change_code(c0, c1):
    """Change category (CHANGES.code) for one class transition c0 -> c1 (class values 1–7)."""
    if c0 == c1:
        return 0
    return {("N", "N"): 1, ("P", "P"): 1, ("O", "N"): 2, ("O", "P"): 2, ("P", "N"): 2,
            ("N", "O"): 3, ("P", "O"): 3, ("N", "P"): 4}[(_group(c0), _group(c1))]

PA_COLS = ["SITE_ID", "NAME", "NAME_ENG", "DESIG_ENG", "DESIG_TYPE", "IUCN_CAT", "REALM", "STATUS", "STATUS_YR",
           "GIS_AREA", "GIS_M_AREA", "LAND_KM2", "LON", "LAT"]


def init(project):
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def class_image(year, region=None):
    """ForTy V2 Beta class map (1–7) for one year as a single byte band named y<year>; nodata stays masked."""
    ic = ee.ImageCollection(f"{FORTY}/{year}")
    if region is not None:
        ic = ic.filterBounds(region)
    raw = ic.mosaic()
    # probabilities are 0–250; 255 marks nodata, and an all-zero pixel has no prediction
    raw = raw.updateMask(raw.select(0).lte(250).And(raw.reduce(ee.Reducer.sum()).gt(0)))
    return raw.toArray().arrayArgmax().arrayGet([0]).add(1).toByte().rename(f"y{year}")


def change_image(y0, y1, region=None):
    """Change category (CHANGES.code, 0–4) per pixel from y0 to y1; masked where either year has no data.

    Same rules as change_code(), written as image algebra.
    """
    c0, c1 = class_image(y0, region), class_image(y1, region)

    def group(c):  # 0 natural, 1 planted / tree crops, 2 other land
        return c.remap([1, 7, 2, 3, 4, 5, 6], [0, 0, 0, 1, 1, 1, 2])

    g0, g1 = group(c0), group(c1)
    code = (ee.Image(0)
            .where(c0.neq(c1).And(g0.eq(g1)), 1)
            .where(g0.eq(2).And(g1.neq(2)).Or(g0.eq(1).And(g1.eq(0))), 2)
            .where(g0.neq(2).And(g1.eq(2)), 3)
            .where(g0.eq(0).And(g1.eq(1)), 4))
    return code.updateMask(c0.mask().And(c1.mask())).toByte().rename("change")


def select_pas(iso3="VNM", status=("Designated", "Inscribed"), desig_types=("National", "International"),
               exclude_realm=("Marine",), min_land_km2=1.0):
    """Protected areas of one country. Returns (ee.FeatureCollection, DataFrame of attributes, sorted by SITE_ID).

    LAND_KM2 = GIS_AREA − GIS_M_AREA (the non-marine part); LON/LAT is the centroid, used to pick the UTM zone.
    """
    fc = (ee.FeatureCollection(WDPA).filter(ee.Filter.eq("ISO3", iso3))
          .filter(ee.Filter.inList("STATUS", list(status)))
          .filter(ee.Filter.inList("DESIG_TYPE", list(desig_types))))
    if exclude_realm:
        fc = fc.filter(ee.Filter.inList("REALM", list(exclude_realm)).Not())

    def _attrs(f):
        c = f.geometry().centroid(100).coordinates()
        return f.set({"LAND_KM2": ee.Number(f.get("GIS_AREA")).subtract(f.get("GIS_M_AREA")),
                      "LON": c.get(0), "LAT": c.get(1)})

    fc = fc.map(_attrs).filter(ee.Filter.gte("LAND_KM2", min_land_km2))
    rows = fc.reduceColumns(ee.Reducer.toList(len(PA_COLS)), PA_COLS).get("list").getInfo()
    df = pd.DataFrame(rows, columns=PA_COLS).sort_values("SITE_ID").reset_index(drop=True)
    df["SITE_ID"] = df.SITE_ID.astype(int)
    return fc, df


def pa_feature(fc, site_id):
    return ee.Feature(fc.filter(ee.Filter.eq("SITE_ID", int(site_id))).first())


def utm_epsg(lon, lat):
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def _pa_areas(fc, pa, years, scales, tile_scale):
    """Class areas (ha) of one PA for every year in one request, in the PA's UTM zone so a pixel is scale² m².

    frequencyHistogram counts edge pixels by the fraction inside the polygon. Coarser scales are tried only when the
    finer one fails (timeout or memory on very large PAs); the scale used is kept with the result.
    """
    geom = pa_feature(fc, pa.SITE_ID).geometry()
    img = ee.Image.cat([class_image(y, geom) for y in years])
    crs = f"EPSG:{utm_epsg(pa.LON, pa.LAT)}"
    errors = []
    for scale in scales:
        try:
            h = img.reduceRegion(ee.Reducer.frequencyHistogram(), geom, scale=scale, crs=crs,
                                 maxPixels=1e13, tileScale=tile_scale).getInfo()
            rows = [{"SITE_ID": int(pa.SITE_ID), "year": y, "value": int(k), "area_ha": n * scale ** 2 / 1e4,
                     "scale_m": scale} for y in years for k, n in (h.get(f"y{y}") or {}).items()]
            return rows, errors
        except Exception as e:
            errors.append(f"scale {scale} m: {type(e).__name__}: {str(e)[:160]}")
    return None, errors


def _pa_transitions(fc, pa, years, scales, tile_scale):
    """Area (ha) of every class transition first year -> last year in one PA, at 10 m in the PA's UTM zone."""
    geom = pa_feature(fc, pa.SITE_ID).geometry()
    y0, y1 = years[0], years[-1]
    pair = class_image(y0, geom).multiply(10).add(class_image(y1, geom)).rename("t")  # 17 = class 1 -> class 7
    crs = f"EPSG:{utm_epsg(pa.LON, pa.LAT)}"
    errors = []
    for scale in scales:
        try:
            h = pair.reduceRegion(ee.Reducer.frequencyHistogram(), geom, scale=scale, crs=crs,
                                  maxPixels=1e13, tileScale=tile_scale).getInfo()
            return [{"SITE_ID": int(pa.SITE_ID), "from": int(k) // 10, "to": int(k) % 10,
                     "area_ha": n * scale ** 2 / 1e4, "scale_m": scale} for k, n in (h.get("t") or {}).items()], errors
        except Exception as e:
            errors.append(f"scale {scale} m: {type(e).__name__}: {str(e)[:160]}")
    return None, errors


def _per_pa(fn, fc, pas, cache_dir, key, what, workers, log, *args):
    """Runs fn(fc, pa, *args) -> (rows, errors) for every PA not yet cached as <cache_dir>/<SITE_ID>_<key>.json.

    Returns (all rows, {SITE_ID: errors} for the PAs that failed at every scale).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out, todo, failed = [], [], {}
    for pa in pas.itertuples():
        f = cache_dir / f"{pa.SITE_ID}_{key}.json"
        if f.exists():
            out += json.loads(f.read_text(encoding="utf-8"))
        else:
            todo.append(pa)
    log(f"{what}: {len(pas) - len(todo)} PA lấy từ cache, {len(todo)} PA cần tính trên GEE")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, fc, pa, *args): pa for pa in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            pa = futs[fut]
            rows, errors = fut.result()
            if rows is None:
                failed[pa.SITE_ID] = errors
                log(f"  [{i}/{len(todo)}] LỖI {pa.SITE_ID} {pa.NAME_ENG}: {errors[-1]}")
                continue
            (cache_dir / f"{pa.SITE_ID}_{key}.json").write_text(json.dumps(rows), encoding="utf-8")
            out += rows
            note = f" (dùng {rows[0]['scale_m']} m)" if errors else ""
            log(f"  [{i}/{len(todo)}] {pa.SITE_ID} {pa.NAME_ENG}{note} – {time.time() - t0:.0f}s")
    return out, failed


def compute_areas(fc, pas, years, cache_dir, scales=(10, 20, 30), tile_scale=4, workers=6, log=print):
    """Long table SITE_ID | year | value | area_ha | scale_m for all PAs, cached as one JSON per PA."""
    out, failed = _per_pa(_pa_areas, fc, pas, cache_dir, f"{min(years)}-{max(years)}", "Diện tích", workers, log,
                          years, scales, tile_scale)
    df = pd.DataFrame(out, columns=["SITE_ID", "year", "value", "area_ha", "scale_m"])
    return df.merge(CLASSES[["value", "label", "label_en"]], on="value", how="left"), failed


def compute_transitions(fc, pas, years, cache_dir, scales=(10, 20, 30), tile_scale=4, workers=6, log=print):
    """Long table SITE_ID | from | to | area_ha | scale_m | change (CHANGES.code), first year -> last year."""
    out, failed = _per_pa(_pa_transitions, fc, pas, cache_dir, f"{years[0]}to{years[-1]}", "Chuyển đổi lớp",
                          workers, log, years, scales, tile_scale)
    df = pd.DataFrame(out, columns=["SITE_ID", "from", "to", "area_ha", "scale_m"])
    df["change"] = [change_code(a, b) for a, b in zip(df["from"], df["to"])]
    return df, failed


def overlaps(fc, pas):
    """Pairs of selected PAs whose polygons intersect: area of the intersection and its share of each PA."""
    def per_a(a):
        others = fc.filterBounds(a.geometry()).filter(ee.Filter.neq("SITE_ID", a.get("SITE_ID")))
        return others.map(lambda b: ee.Feature(None, {
            "a": a.get("SITE_ID"), "b": b.get("SITE_ID"),
            "km2": a.geometry().intersection(b.geometry(), ee.ErrorMargin(50)).area(50).divide(1e6)}))

    rows = fc.map(per_a).flatten().reduceColumns(ee.Reducer.toList(3), ["a", "b", "km2"]).get("list").getInfo()
    df = pd.DataFrame(rows, columns=["a", "b", "km2"]).astype({"a": int, "b": int})
    df = df[(df.a < df.b) & (df.km2 > 0.01)]
    info = pas.set_index("SITE_ID")
    df["name_a"], df["name_b"] = info.NAME.reindex(df.a).values, info.NAME.reindex(df.b).values
    df["desig_a"], df["desig_b"] = info.DESIG_ENG.reindex(df.a).values, info.DESIG_ENG.reindex(df.b).values
    # GIS_AREA is WDPA's own figure and can be slightly below the geometry's area, hence the clip at 100 %
    df["pct_of_a"] = (100 * df.km2 / info.GIS_AREA.reindex(df.a).values).clip(upper=100)
    df["pct_of_b"] = (100 * df.km2 / info.GIS_AREA.reindex(df.b).values).clip(upper=100)
    return df.sort_values("km2", ascending=False).reset_index(drop=True)


def _region(pa, geom):
    pad = max(300, (pa.GIS_AREA ** 0.5) * 1000 * 0.04)  # ~4 % of the PA's width around it
    return geom.bounds(100).buffer(pad, 100).bounds(100)


def _thumb(fc, pa, layer, dims):
    """Thumbnail URL of a visualized layer(geom) clipped to the PA, on white, with the PA outline."""
    f = pa_feature(fc, pa.SITE_ID)
    geom = f.geometry()
    bg = ee.Image(1).visualize(palette=["FFFFFF"])
    outline = ee.FeatureCollection([f]).style(color="333333", fillColor="00000000", width=1)
    img = ee.ImageCollection([bg, layer(geom).clip(geom), outline]).mosaic()
    return img.getThumbURL({"region": _region(pa, geom), "dimensions": dims, "crs": "EPSG:3857", "format": "png"})


def _class_layer(year):
    palette = [CLASSES.set_index("value").color[v].lstrip("#") for v in range(1, 8)]
    return lambda geom: class_image(year, geom).visualize(min=1, max=7, palette=palette)


OUTSIDE, OUTLINE = 255, 254  # codes in the change map array besides CHANGES.code


def _change_codes(geom, f, y0, y1, utm, scale, pixel_m):
    """Change map at `scale` m shrunk to map pixels of pixel_m, as two bands:
    c = most common change code among the changed pixels under the map pixel (0 if none changed), OUTSIDE, OUTLINE;
    f = share of pixels under it that changed, 0–250.
    Shrinking goes in 4×4 steps (10 → 40 → 160 m …): one step from 10 m to a 200 m map pixel exceeds Earth Engine's
    memory limit on large PAs."""
    chg = change_image(y0, y1, geom).reproject(utm.atScale(scale))
    code, frac, s = chg.updateMask(chg.gt(0)), chg.gt(0).toFloat(), scale

    def shrink(code, frac):
        return (code.reduceResolution(ee.Reducer.mode(), maxPixels=64),
                frac.reduceResolution(ee.Reducer.mean(), maxPixels=64))

    while s * 4 <= pixel_m:
        s *= 4
        code, frac = (b.reproject(utm.atScale(s)) for b in shrink(code, frac))
    if s < pixel_m:  # last, partial step into the requested grid
        code, frac = shrink(code, frac)
    outline = ee.Image(0).paint(ee.FeatureCollection([f]), 1, 1).eq(1)
    code = code.unmask(0).clip(geom).unmask(OUTSIDE, False).where(outline, OUTLINE).toByte().rename("c")
    frac = frac.multiply(250).round().unmask(0, False).toByte().rename("f")
    return code.addBands(frac)


def change_array(fc, pa, years, dims=900, scale=10, tile_km2=250, workers=8):
    """Change map first -> last year of one PA in EPSG:3857 as two uint8 arrays (see _change_codes):
    code (CHANGES.code, OUTSIDE, OUTLINE) and share of changed 10 m pixels (0–250).

    A single request for a large PA at 10 m exceeds Earth Engine's memory limit, so the map is fetched with
    ee.data.computePixels in tiles of about tile_km2 on a fixed grid and stitched here.
    """
    f = pa_feature(fc, pa.SITE_ID)
    geom = f.geometry()
    ring = _region(pa, geom).coordinates().getInfo()[0]
    lon0, lat0 = map(min, zip(*ring))
    lon1, lat1 = map(max, zip(*ring))
    R = 6378137.0

    def merc(lon, lat):
        return R * math.radians(lon), R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))

    (x0, y0), (x1, y1) = merc(lon0, lat0), merc(lon1, lat1)
    px = max(x1 - x0, y1 - y0) / dims  # mercator metres per map pixel
    ground = px * math.cos(math.radians((lat0 + lat1) / 2))
    W, H = math.ceil((x1 - x0) / px), math.ceil((y1 - y0) / px)
    img = _change_codes(geom, f, years[0], years[-1], ee.Projection(f"EPSG:{utm_epsg(pa.LON, pa.LAT)}"),
                        scale, ground)
    t = max(16, int(math.sqrt(tile_km2) * 1000 / ground))  # tile side in map pixels
    out = np.full((H, W), OUTSIDE, dtype=np.uint8)
    share = np.zeros((H, W), dtype=np.uint8)

    def _tile(i, j):
        w, h = min(t, W - j), min(t, H - i)
        req = {"expression": img, "fileFormat": "NUMPY_NDARRAY",
               "grid": {"dimensions": {"width": w, "height": h}, "crsCode": "EPSG:3857",
                        "affineTransform": {"scaleX": px, "shearX": 0, "translateX": x0 + j * px,
                                            "shearY": 0, "scaleY": -px, "translateY": y1 - i * px}}}
        for attempt in range(3):
            try:
                a = ee.data.computePixels(req)
                out[i:i + h, j:j + w], share[i:i + h, j:j + w] = a["c"], a["f"]
                return
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(3 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(lambda ij: _tile(*ij), [(i, j) for i in range(0, H, t) for j in range(0, W, t)]))
    return out, share


def render_change(code, share):
    """RGB array: unchanged pixels light gray, changed ones in their CHANGES color blended over the gray with opacity
    sqrt(share), so a map pixel where a few 10 m pixels changed is faint and one where all changed is full color."""
    rgb = {c.code: np.array([int(c.color[k:k + 2], 16) for k in (1, 3, 5)], float) for c in CHANGES.itertuples()}
    out = np.full(code.shape + (3,), 255, dtype=float)
    base = rgb[0]
    inside = code <= CHANGES.code.max()
    out[inside] = base
    alpha = np.sqrt(share.astype(float) / 250)[..., None]
    for k, col in rgb.items():
        if k == 0:
            continue
        m = code == k
        out[m] = base + alpha[m] * (col - base)
    out[code == OUTLINE] = (0x33, 0x33, 0x33)
    return out.round().astype(np.uint8)


def _download(jobs, log, what, workers=6, retries=3):
    """jobs: [(key, path, url_fn)]; fetches each missing PNG (url_fn() builds the URL). Returns {key: Path}."""
    def _get(path, url_fn):
        if path.exists():
            return path, None
        err = None
        for attempt in range(retries):
            try:
                r = requests.get(url_fn(), timeout=300)
                if not r.ok:  # Earth Engine puts the reason (e.g. "User memory limit exceeded.") in the body
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                path.write_bytes(r.content)
                return path, None
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:200]}"
                time.sleep(3 * (attempt + 1))
        return None, err

    done = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_get, path, fn): key for key, path, fn in jobs}
        for fut in as_completed(futs):
            path, err = fut.result()
            if path:
                done[futs[fut]] = path
            else:
                log(f"  LỖI {what} {futs[fut]}: {err}")
    log(f"{what}: {len(done)}/{len(jobs)}")
    return done


def download_thumbs(fc, pas, years, out_dir, dims=900, workers=6, retries=3, log=print):
    """PNG of the class map for each PA and year (non-forest in light gray, PA outline in dark gray); cached on disk.

    Returns {(SITE_ID, year): Path} for the images that exist.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [((pa.SITE_ID, y), out_dir / f"{pa.SITE_ID}_{y}.png",
             lambda pa=pa, y=y: _thumb(fc, pa, _class_layer(y), dims)) for pa in pas.itertuples() for y in years]
    return _download(jobs, log, "Ảnh bản đồ", workers, retries)


def download_change_maps(fc, pas, years, out_dir, dims=900, scale=10, log=print):
    """Change map first -> last year for each PA as a colored PNG (see change_array); cached on disk.

    Returns {SITE_ID: Path}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done, t0 = {}, time.time()
    todo = [pa for pa in pas.itertuples()]
    for i, pa in enumerate(todo, 1):
        path = out_dir / f"{pa.SITE_ID}_change_{years[0]}_{years[-1]}.png"
        if not path.exists():
            try:
                Image.fromarray(render_change(*change_array(fc, pa, years, dims, scale))).save(path, optimize=True)
                log(f"  [{i}/{len(todo)}] {pa.SITE_ID} {pa.NAME_ENG} – {time.time() - t0:.0f}s")
            except Exception as e:
                log(f"  [{i}/{len(todo)}] LỖI {pa.SITE_ID} {pa.NAME_ENG}: {type(e).__name__}: {str(e)[:200]}")
                continue
        done[pa.SITE_ID] = path
    log(f"Bản đồ thay đổi: {len(done)}/{len(todo)}")
    return done
