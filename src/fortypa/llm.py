"""Short observation of forest type changes per PA, written by Gemini from the areas this pipeline computed.

The model only receives the computed numbers and is told to use nothing else. Answers are cached by (model, prompt)
under cache_dir, so re-running on the same data does not call the API again. Without an API key, or when the call
fails, observation() returns (None, reason, usage) and the report says so.
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path

import pandas as pd

from .gee import FOREST, NATURAL

SYSTEM = """Bạn là chuyên gia lâm nghiệp viết nhận xét ngắn về biến động rừng trong khu bảo tồn từ bản đồ phân loại rừng vệ tinh.
Quy tắc bắt buộc:
- Chỉ dùng số liệu được cung cấp. Không bịa số, sự kiện, chính sách hay địa danh.
- Nguyên nhân chỉ nêu dưới dạng khả năng ("có thể do…") và chỉ khi số liệu gợi ý rõ; không khẳng định.
- Dữ liệu là ForTy V2 Beta (thử nghiệm): chênh lệch nhỏ giữa các năm có thể là nhiễu phân loại, không phải thay đổi thật. Phân biệt xu hướng ổn định nhiều năm với dao động một năm.
- Viết tiếng Việt, văn xuôi, một đoạn 3–5 câu, 70–130 từ. Không tiêu đề, không gạch đầu dòng, không Markdown.
- Làm tròn số (ví dụ khoảng 1.250 ha, giảm 4,2%), dùng dấu chấm phân cách hàng nghìn và dấu phẩy thập phân kiểu Việt Nam.
"""

TASK = """Viết nhận xét ngắn về biến động các loại rừng của khu bảo tồn dưới đây giai đoạn {y0}–{y1}: \
tình trạng chung và tỷ lệ rừng tự nhiên, các lớp thay đổi nhiều nhất (ha và %), xu hướng đều hay dao động, \
và một điểm cần lưu ý hoặc kiểm tra thực địa.

Dữ liệu (JSON, tính từ bản đồ ForTy V2 Beta 10 m):
"""


def _r(v, d=1):
    return None if v is None or v != v else round(float(v), d)


def pa_facts(pa, areas_pa, years):
    """Compact, JSON-able facts about one PA for the prompt."""
    y0, y1 = years[0], years[-1]
    t = (areas_pa.pivot_table(index="year", columns="value", values="area_ha", aggfunc="sum")
         .reindex(index=list(years)).fillna(0))
    analysed = t.sum(axis=1)
    forest = t[[v for v in t.columns if v != 6]].sum(axis=1)
    natural = t[[v for v in NATURAL if v in t.columns]].sum(axis=1)
    per_class = {}
    for c in FOREST.itertuples():
        s = t[c.value] if c.value in t.columns else pd.Series(0.0, index=t.index)
        if s.max() < 0.5:
            continue
        d = s[y1] - s[y0]
        per_class[c.label] = {"theo_nam_ha": {int(y): _r(s[y], 0) for y in years},
                              "thay_doi_ha": _r(d, 0), "thay_doi_pct": _r(100 * d / s[y0]) if s[y0] > 0 else None}
    return {
        "ten": pa.NAME, "ten_tieng_anh": pa.NAME_ENG, "loai_khu": pa.DESIG_ENG, "cap": pa.DESIG_TYPE,
        "iucn": pa.IUCN_CAT, "nam_cong_nhan": int(pa.STATUS_YR) if pa.STATUS_YR else None,
        "dien_tich_wdpa_ha": _r(pa.GIS_AREA * 100, 0), "dien_tich_phan_tich_ha": _r(analysed.mean(), 0),
        "giai_doan": f"{y0}–{y1}",
        "tong_dien_tich_rung_va_cay_ha": {int(y): _r(forest[y], 0) for y in years},
        "ty_le_rung_tu_nhien_pct": {int(y0): _r(100 * natural[y0] / analysed[y0]) if analysed[y0] else None,
                                    int(y1): _r(100 * natural[y1] / analysed[y1]) if analysed[y1] else None},
        "dat_khac_khong_phai_rung_ha": {int(y0): _r(t.get(6, pd.Series(0, index=t.index))[y0], 0),
                                        int(y1): _r(t.get(6, pd.Series(0, index=t.index))[y1], 0)},
        "theo_loai_rung": per_class,
        "ghi_chu": "Rừng tự nhiên = nguyên sinh trưởng thành + nguyên sinh đang phục hồi + tái sinh tự nhiên.",
    }


def build_prompt(facts, years):
    return TASK.format(y0=years[0], y1=years[-1]) + json.dumps(facts, ensure_ascii=False, indent=1)


def latest_flash(api_key_env="GEMINI_API_KEY", fallback="gemini-3.8-flash"):
    """Highest-version plain text 'gemini-X.Y-flash' model the key can use (no -lite / -tts / -image variants)."""
    try:
        from google import genai
        names = [m.name.split("/")[-1] for m in genai.Client(api_key=os.environ[api_key_env]).models.list()]
        found = [(tuple(int(p) for p in m.group(1).split(".")), n) for n in names
                 if (m := re.fullmatch(r"gemini-(\d+(?:\.\d+)?)-flash", n))]
        return max(found)[1] if found else fallback
    except Exception:
        return fallback


def _usage(resp):
    u = getattr(resp, "usage_metadata", None)
    return {k: getattr(u, f, None) for k, f in (("input", "prompt_token_count"), ("output", "candidates_token_count"),
                                                ("thinking", "thoughts_token_count"), ("total", "total_token_count"))}


_daily_quota_hit = {}  # model -> error message, once the per-day quota is exhausted (no point calling again today)


def observation(facts, years, model, cache_dir, api_key_env="GEMINI_API_KEY", retries=4, wait_s=5):
    """Returns (text, None, usage) on success or (None, reason, usage).

    Server errors (500/503), rate limits (429) and network errors are retried `retries` times, waiting wait_s,
    2*wait_s, 4*wait_s…
    """
    usage = {"source": "none", "attempts": 0, **_usage(None)}
    prompt = build_prompt(facts, years)
    key = hashlib.sha256(f"{model}\n{SYSTEM}\n{prompt}".encode()).hexdigest()[:20]
    cache = Path(cache_dir) / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))["text"], None, {**usage, "source": "cache"}
    if model in _daily_quota_hit:
        return None, _daily_quota_hit[model], usage
    api_key = os.environ.get(api_key_env)
    if not api_key:
        return None, f"chưa đặt biến môi trường {api_key_env}", usage
    from google import genai
    from google.genai import errors, types

    client = genai.Client(api_key=api_key)
    for attempt in range(retries + 1):
        usage["attempts"] = attempt + 1
        try:
            resp = client.models.generate_content(model=model, contents=prompt,
                                                  config=types.GenerateContentConfig(system_instruction=SYSTEM))
            text = " ".join(p.strip() for p in (resp.text or "").splitlines() if p.strip())
            break
        except Exception as e:
            # client errors other than 429 (bad key, unknown model…) won't fix themselves; server errors, rate limits
            # and dropped connections usually do
            permanent = isinstance(e, errors.ClientError) and getattr(e, "code", None) != 429
            if getattr(e, "code", None) == 429 and "per day" in str(e).lower():
                _daily_quota_hit[model] = (f"hết quota trong ngày của {model} (gói miễn phí); chạy lại notebook sau "
                                           f"khi quota được làm mới, các khu đã có nhận xét không gọi lại")
                return None, _daily_quota_hit[model], usage
            if permanent or attempt == retries:
                return None, f"lỗi gọi {model}: {type(e).__name__}: {str(e)[:200]}", usage
            time.sleep(wait_s * 2 ** attempt)
    usage.update(_usage(resp), source="api")
    if not text:
        return None, f"{model} không trả về nội dung", usage
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"model": model, "created": pd.Timestamp.now().isoformat(timespec="seconds"),
                                 "usage": _usage(resp), "prompt": prompt, "text": text}, ensure_ascii=False, indent=1),
                     encoding="utf-8")
    return text, None, usage
