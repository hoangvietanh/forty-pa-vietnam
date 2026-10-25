"""HTML report: method and legend, summary of all PAs, overlaps, then one section per PA.

Images are either embedded (one self-contained file) or written next to the page in img/ (e.g. for GitHub Pages)."""
import base64
import html
import io
from pathlib import Path

import pandas as pd
from PIL import Image

from .charts import area_table, fmt_ha
from .gee import CHANGES, CLASSES, FOREST, NATURAL

CSS = """
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#1f1f1f;--muted:#6b6b6b;--line:#e6e6e3;--accent:#16733a;--warn:#8a5a00}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:20px;margin:36px 0 12px}h3{font-size:18px;margin:0 0 4px}
.card>h3{font-size:16px;margin:4px 0 8px;color:var(--accent)}
.sub{color:var(--muted);margin:0 0 20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;margin:16px 0}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600;background:#f6f6f4;position:sticky;top:0}
td.n,th.n{text-align:right}
.scroll{overflow-x:auto}
img{max-width:100%;height:auto;display:block;margin:8px 0}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;border:1px solid #bdbdbd;vertical-align:-1px;margin-right:6px}
.ai{background:#f3f8f4;border-left:3px solid var(--accent);padding:10px 14px;border-radius:6px;margin:10px 0}
.ai .tag{font-size:12px;color:var(--muted);display:block;margin-bottom:4px}
.note{color:var(--warn)}
.meta{color:var(--muted);font-size:13px}
details summary{cursor:pointer;color:var(--muted);font-size:13px;margin-top:6px}
a{color:var(--accent)}
.up{color:#1b6e3a}.down{color:#b0303a}
"""


def _png_256(path):
    """PNG bytes reduced to a 256-color palette: the maps and charts are flat colors, so this is ~4x smaller."""
    buf = io.BytesIO()
    Image.open(path).convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT).save(buf, "PNG", optimize=True)
    return buf.getvalue()


def _img(path, embed, base_dir, alt=""):
    """Embedded as base64, or copied (256-color) to <report dir>/img/ and linked, lazy-loaded, so the page itself
    stays small and images load as the reader scrolls (e.g. for GitHub Pages)."""
    path = Path(path)
    alt = html.escape(alt, quote=True)
    if embed:
        return f'<img alt="{alt}" src="data:image/png;base64,{base64.b64encode(_png_256(path)).decode()}">'
    img_dir = Path(base_dir) / "img"
    img_dir.mkdir(parents=True, exist_ok=True)
    dst = img_dir / path.name
    if not dst.exists() or dst.stat().st_mtime < path.stat().st_mtime:
        dst.write_bytes(_png_256(path))
    w, h = Image.open(path).size  # width/height let the browser reserve space before the image arrives
    return f'<img alt="{alt}" src="img/{html.escape(dst.name)}" width="{w}" height="{h}" loading="lazy">'


def _pct(a, b):
    return None if not a else 100 * (b - a) / a


def _pct_html(p, d=1):
    if p is None or p != p:
        return "–"
    cls = "up" if p > 0.05 else "down" if p < -0.05 else ""
    return f'<span class="{cls}">{"+" if p > 0 else ""}{fmt_ha(p, d)}%</span>'


def summary_table(pas, areas, years, transitions=None):
    """One row per PA: forest (all tree classes) and natural forest area, first and last year, and the share of the
    analysed area whose class changed between them."""
    y0, y1 = years[0], years[-1]
    rows = []
    for pa in pas.itertuples():
        a = areas[areas.SITE_ID == pa.SITE_ID]
        if a.empty:
            continue
        changed_pct = None
        if transitions is not None:
            t = transitions[transitions.SITE_ID == pa.SITE_ID]
            if t.area_ha.sum() > 0:
                changed_pct = 100 * t[t.change > 0].area_ha.sum() / t.area_ha.sum()
        by = a.pivot_table(index="year", columns="value", values="area_ha", aggfunc="sum").reindex(list(years)).fillna(0)
        forest = by[[v for v in by.columns if v != 6]].sum(axis=1)
        natural = by[[v for v in NATURAL if v in by.columns]].sum(axis=1)
        rows.append({"SITE_ID": pa.SITE_ID, "NAME": pa.NAME, "DESIG_ENG": pa.DESIG_ENG, "IUCN_CAT": pa.IUCN_CAT,
                     "area_ha": pa.GIS_AREA * 100, f"forest_{y0}": forest[y0], f"forest_{y1}": forest[y1],
                     "forest_change_pct": _pct(forest[y0], forest[y1]),
                     f"natural_{y0}": natural[y0], f"natural_{y1}": natural[y1],
                     "natural_change_pct": _pct(natural[y0], natural[y1]), "scale_m": a.scale_m.max(),
                     "changed_pct": changed_pct})
    return pd.DataFrame(rows)


def _transitions_html(t, years, top=8):
    """The largest class transitions of one PA (from -> to, ha, % of the analysed area)."""
    lab = CLASSES.set_index("value").label
    total = t.area_ha.sum()
    ch = t[t.change > 0].rename(columns={"from": "src"}).sort_values("area_ha", ascending=False).head(top)
    if ch.empty:
        return "<p class='meta'>Không có pixel nào đổi lớp.</p>"
    sw = CHANGES.set_index("code").color
    rows = "".join(f"<tr><td><span class='sw' style='background:{sw[r.change]}'></span>{html.escape(lab[r.src])}</td>"
                   f"<td>{html.escape(lab[r.to])}</td><td class='n'>{fmt_ha(r.area_ha)}</td>"
                   f"<td class='n'>{fmt_ha(100 * r.area_ha / total, 2)}%</td></tr>"
                   for r in ch.itertuples())
    return (f"<details><summary>Các chuyển đổi lớp lớn nhất {years[0]} → {years[-1]}</summary>"
            f"<div class='scroll'><table><tr><th>Từ lớp</th><th>Sang lớp</th><th class='n'>Diện tích (ha)</th>"
            f"<th class='n'>% diện tích</th></tr>{rows}</table></div></details>")


def build_report(out_path, pas, areas, years, figures, observations, overlaps_df=None, failed=None, model=None,
                 embed=True, transitions=None, title="Biến động loại rừng trong các khu bảo tồn Việt Nam"):
    """figures: {SITE_ID: {"map": png, "change": png, "chart": png}}; observations: {SITE_ID: (text, reason)};
    transitions: long table from gee.compute_transitions (optional)."""
    out_path = Path(out_path)
    base = out_path.parent
    y0, y1 = years[0], years[-1]
    s = summary_table(pas, areas, years, transitions)
    p = [f"<!doctype html><html lang='vi'><head><meta charset='utf-8'>"
         f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>"
         f"<style>{CSS}</style></head><body><main>",
         f"<h1>{html.escape(title)}</h1>",
         f"<p class='sub'>ForTy V2 Beta {y0}–{y1} · {len(s)} khu bảo tồn (WDPA) · tạo ngày "
         f"{pd.Timestamp.now():%d/%m/%Y %H:%M}</p>"]

    legend = "".join(f"<tr><td><span class='sw' style='background:{c.color}'></span>{html.escape(c.label)}</td>"
                     f"<td>{html.escape(c.label_en)}</td><td class='n'>{c.value}</td><td>{c.band}</td></tr>"
                     for c in CLASSES.itertuples())
    p.append(f"""<div class='card'><h2 style='margin-top:0'>Dữ liệu và phương pháp</h2>
<h3>Dữ liệu</h3>
<p><b>Bộ dữ liệu Forest Typology (ForTy).</b> Dữ liệu chính của phân tích là bộ Forest Typology (ForTy) do Google DeepMind
và Google Research phát triển cùng Viện Tài nguyên Thế giới (WRI) và Viện Phân tích Hệ thống Ứng dụng Quốc tế (IIASA),
công bố tháng 10/2026. ForTy phân loại từng pixel 10 m theo định nghĩa của FAO thành: rừng nguyên sinh trưởng thành; rừng
nguyên sinh đang phục hồi; rừng tái sinh tự nhiên; rừng trồng quản lý theo chu kỳ dài (planted forest); rừng trồng thâm
canh, sinh trưởng nhanh, thường thuần loài (plantation forest); cây công nghiệp lâu năm và nông lâm kết hợp (cao su, cà
phê, ca cao, cây ăn quả...); và đất khác. Mô hình học sâu sử dụng đầu vào chính là embedding AlphaEarth Foundations – dạng
tóm tắt nhiều năm quan sát vệ tinh cho mỗi pixel 10 m – nhờ đó nhận diện được các dấu hiệu như kiểu bố trí trồng, chu kỳ
sinh trưởng và nhịp khai thác; tập mẫu huấn luyện gồm khoảng 1,7 triệu điểm. Bản đồ năm 2020 được kiểm định tại hơn 9.000
điểm chọn ngẫu nhiên trên toàn cầu, đạt độ chính xác 90% về loại rừng và 95% về phân biệt rừng/không rừng. Chuỗi thời
gian {y0}–{y1} dùng trong phân tích này là bản phát hành sớm, áp dụng cách tiếp cận tương tự cho từng năm; độ chính xác
nêu trên thuộc về bản 2020, chưa có đánh giá riêng cho chuỗi thời gian cũng như cho Việt Nam, nên kết quả thay đổi lớp cần
được hiểu là ước tính. Sản phẩm cung cấp xác suất cho từng lớp thay vì nhãn cứng; phân tích này chọn lớp có xác suất cao
nhất, không áp ngưỡng tin cậy.</p>
<p><b>Dữ liệu trên Google Earth Engine.</b> <code>projects/nature-trace/assets/forest_typology/v2_beta/&lt;năm&gt;</code>
(ForTy V2 Beta), mỗi năm một bộ ảnh 10 m gồm 7 band xác suất (0–250), tương ứng 7 lớp trong bảng dưới.</p>
<p><b>Khu bảo tồn.</b> <code>WCMC/WDPA/current/polygons</code>, ISO3 = VNM, trạng thái Designated/Inscribed, không gồm khu
thuần biển và khu nhỏ hơn 1 km². Các khu quốc gia và quốc tế (UNESCO-MAB, Ramsar, Di sản) đều được tính riêng nên một
vùng rừng có thể thuộc nhiều khu (xem mục Chồng lấn).</p>
<div class='scroll'><table><tr><th>Lớp</th><th>Tên gốc</th><th class='n'>Giá trị</th><th>Band</th></tr>{legend}</table></div>
<h3 style='margin-top:18px'>Phương pháp</h3>
<p><b>Diện tích theo năm.</b> Với mỗi khu và mỗi năm {y0}–{y1}, diện tích từng lớp được tính trong hệ UTM của khu ở độ
phân giải 10 m (mỗi pixel 100 m²); pixel ở mép ranh giới được tính theo phần nằm trong khu. “Rừng &amp; cây” là mọi lớp trừ
Đất khác; “rừng tự nhiên” là nguyên sinh trưởng thành, nguyên sinh đang phục hồi và tái sinh tự nhiên.</p>
<p><b>Bản đồ thay đổi {y0} → {y1}.</b> So lớp của từng pixel 10 m giữa hai năm, gộp thành 4 loại thay đổi: rừng tự nhiên →
rừng trồng / cây công nghiệp; rừng / cây → đất khác; tăng rừng hoặc phục hồi (đất khác → rừng, rừng trồng → tự nhiên); đổi
lớp trong cùng nhóm. Diện tích mỗi loại và từng cặp lớp tính ở 10 m. Đây là thay đổi gộp cả hai chiều nên thường lớn hơn
thay đổi ròng trên biểu đồ. Trên bản đồ, mỗi điểm ảnh đại diện cho nhiều pixel 10 m: màu là loại thay đổi chiếm ưu thế
trong điểm ảnh đó, độ đậm là tỷ lệ pixel 10 m đổi lớp, nên vùng chỉ lác đác vài pixel đổi lớp hiện nhạt.</p>
<p><b>Nhận xét tự động.</b> Đoạn nhận xét của mỗi khu do mô hình AI {html.escape(model or 'Gemini')} viết, chỉ từ bảng diện
tích theo năm của khu đó; mô hình được yêu cầu không dùng thông tin ngoài bảng và chỉ nêu nguyên nhân dưới dạng khả năng.</p>
<p class='note'><b>Lưu ý:</b> V2 Beta là dữ liệu thử nghiệm; thứ tự band và định nghĩa lớp có thể thay đổi trước bản phát
hành chính thức. Chênh lệch nhỏ giữa các năm, nhất là ở vùng giáp ranh rừng và đất khác, có thể là nhiễu phân loại. Kết
quả và nhận xét AI cần được đối chiếu với thực địa.</p>
<p><b>Tài liệu tham khảo</b></p>
<ul>
<li>Neumann, M., Raichuk, A., Potapov, P. et al. Global forest typology at 10-meter resolution for forest and land-use
monitoring. <i>Sci Data</i> 13, 1434 (2026).
<a href='https://doi.org/10.1038/s41597-026-08299-5'>https://doi.org/10.1038/s41597-026-08299-5</a></li>
<li>Google DeepMind. Beyond tree cover: mapping the world’s forest types.
<a href='https://deepmind.google/blog/beyond-tree-cover-mapping-the-worlds-forest-types/'>https://deepmind.google/blog/beyond-tree-cover-mapping-the-worlds-forest-types/</a></li>
<li>UNEP-WCMC and IUCN ({pd.Timestamp.now().year}). Protected Planet: The World Database on Protected Areas (WDPA).
Cambridge, UK: UNEP-WCMC and IUCN. <a href='https://www.protectedplanet.net'>www.protectedplanet.net</a>
(truy cập qua Google Earth Engine, <code>WCMC/WDPA/current/polygons</code>, {pd.Timestamp.now():%m/%Y}).</li>
</ul>
</div>""")

    # summary
    hdr = (f"<tr><th>#</th><th>Khu bảo tồn</th><th>Loại</th><th>IUCN</th><th class='n'>Diện tích WDPA (ha)</th>"
           f"<th class='n'>Rừng & cây {y0} (ha)</th><th class='n'>{y1} (ha)</th><th class='n'>Thay đổi</th>"
           f"<th class='n'>Rừng tự nhiên {y0} (ha)</th><th class='n'>{y1} (ha)</th><th class='n'>Thay đổi</th>"
           f"<th class='n'>Đổi lớp {y0}→{y1}</th></tr>")
    body = "".join(
        f"<tr><td>{i}</td><td><a href='#pa{r.SITE_ID}'>{html.escape(r.NAME)}</a></td><td>{html.escape(r.DESIG_ENG)}</td>"
        f"<td>{html.escape(str(r.IUCN_CAT))}</td><td class='n'>{fmt_ha(r.area_ha)}</td>"
        f"<td class='n'>{fmt_ha(r[6])}</td><td class='n'>{fmt_ha(r[7])}</td><td class='n'>{_pct_html(r.forest_change_pct)}</td>"
        f"<td class='n'>{fmt_ha(r[9])}</td><td class='n'>{fmt_ha(r[10])}</td><td class='n'>{_pct_html(r.natural_change_pct)}</td>"
        f"<td class='n'>{'–' if r.changed_pct is None or r.changed_pct != r.changed_pct else fmt_ha(r.changed_pct, 1) + '%'}</td></tr>"
        for i, r in enumerate(s.itertuples(), 1))
    p.append(f"<h2>Tổng hợp</h2><p class='meta'>“Rừng &amp; cây” = mọi lớp trừ Đất khác. “Rừng tự nhiên” = nguyên sinh "
             f"trưởng thành + nguyên sinh đang phục hồi + tái sinh tự nhiên. “Đổi lớp” = phần diện tích có lớp năm {y1} "
             f"khác năm {y0} (tính ở 10 m, gồm cả thay đổi hai chiều nên thường lớn hơn thay đổi ròng).</p>"
             f"<div class='card scroll'><table>{hdr}{body}</table></div>")

    if overlaps_df is not None and len(overlaps_df):
        rows = "".join(f"<tr><td>{html.escape(str(o.name_a))}<div class='meta'>{html.escape(str(o.desig_a))}</div></td>"
                       f"<td>{html.escape(str(o.name_b))}<div class='meta'>{html.escape(str(o.desig_b))}</div></td>"
                       f"<td class='n'>{fmt_ha(o.km2 * 100)}</td><td class='n'>{fmt_ha(o.pct_of_a, 1)}%</td>"
                       f"<td class='n'>{fmt_ha(o.pct_of_b, 1)}%</td></tr>" for o in overlaps_df.itertuples())
        p.append(f"<h2>Chồng lấn giữa các khu</h2><p class='meta'>{len(overlaps_df)} cặp khu có ranh giới giao nhau. "
                 f"Phần giao được tính trong cả hai khu.</p><div class='card scroll'><table><tr><th>Khu A</th><th>Khu B</th>"
                 f"<th class='n'>Diện tích giao (ha)</th><th class='n'>% của A</th><th class='n'>% của B</th></tr>"
                 f"{rows}</table></div>")
    if failed:
        p.append("<h2>Khu chưa tính được</h2><div class='card'><ul>" + "".join(
            f"<li>{k}: {html.escape(v[-1] if isinstance(v, list) else str(v))}</li>" for k, v in failed.items())
                 + "</ul></div>")

    # one section per PA
    p.append("<h2>Chi tiết từng khu</h2>")
    meta = FOREST.set_index("value")
    for i, pa in enumerate(pas.itertuples(), 1):
        a = areas[areas.SITE_ID == pa.SITE_ID]
        if a.empty:
            continue
        f = figures.get(pa.SITE_ID, {})
        text, reason = observations.get(pa.SITE_ID, (None, "chưa chạy"))
        t = area_table(a, years)
        cols = [v for v in t.columns if t[v].sum() > 0]
        thead = "".join(f"<th class='n'><span class='sw' style='background:{meta.color[v]}'></span>"
                        f"{html.escape(meta.label[v])}</th>" for v in cols)
        trows = "".join(f"<tr><td>{y}</td>" + "".join(f"<td class='n'>{fmt_ha(t.loc[y, v])}</td>" for v in cols)
                        + f"<td class='n'><b>{fmt_ha(t.loc[y, cols].sum())}</b></td></tr>" for y in years)
        coarse = a.scale_m.max()
        p.append(f"<section class='card' id='pa{pa.SITE_ID}'><h3>{i}. {html.escape(pa.NAME)}</h3>"
                 f"<p class='meta'>{html.escape(pa.NAME_ENG)} · {html.escape(pa.DESIG_ENG)} ({html.escape(pa.DESIG_TYPE)})"
                 f" · IUCN {html.escape(str(pa.IUCN_CAT))} · công nhận {pa.STATUS_YR} · WDPA {pa.SITE_ID}"
                 f" · {fmt_ha(pa.GIS_AREA * 100)} ha"
                 + (f" · <span class='note'>diện tích tính ở {coarse} m</span>" if coarse > 10 else "") + "</p>")
        if text:
            p.append(f"<div class='ai'><span class='tag'>Nhận xét do AI viết ({html.escape(model or '')})</span>"
                     f"{html.escape(text)}</div>")
        else:
            p.append(f"<p class='meta'>Chưa có nhận xét AI: {html.escape(str(reason))}</p>")
        if f.get("map"):
            p.append(_img(f["map"], embed, base, f"Bản đồ loại rừng {pa.NAME} năm {y0} và {y1}"))
        if f.get("change"):
            p.append(_img(f["change"], embed, base, f"Bản đồ thay đổi loại rừng {pa.NAME} {y0}–{y1}"))
        if transitions is not None:
            p.append(_transitions_html(transitions[transitions.SITE_ID == pa.SITE_ID], years))
        if f.get("chart"):
            p.append(_img(f["chart"], embed, base, f"Biểu đồ diện tích theo loại rừng {pa.NAME} {y0}–{y1}"))
        p.append(f"<details><summary>Bảng số liệu (ha)</summary><div class='scroll'><table><tr><th>Năm</th>{thead}"
                 f"<th class='n'>Tổng</th></tr>{trows}</table></div></details></section>")
    p.append("</main></body></html>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(p), encoding="utf-8")
    return out_path
