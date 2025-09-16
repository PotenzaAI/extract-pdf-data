# =========================================================
# PDF → Markdown (Docling) + Upload para Supabase
# Estrutura limpa, sem prints desnecessários (usa logging).
# =========================================================

import argparse
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
from PIL import Image
from dotenv import load_dotenv
from supabase import create_client
from urllib.parse import urlparse

import cv2  # OpenCV
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

# Fallback extractor
try:
    import pymupdf4llm  # pip install pymupdf4llm

    HAS_PYMUPDF4LLM = True
except Exception:
    HAS_PYMUPDF4LLM = False

# =========================================================
# Logging
# =========================================================

LOGGER = logging.getLogger("pdf_to_md")

def setup_logging(level: str = "WARNING") -> None:
    lvl = getattr(logging, level.upper(), logging.WARNING)
    logging.basicConfig(
        level=lvl,
        format="%(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    LOGGER.setLevel(lvl)

# =========================================================
# Utils
# =========================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def slug_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_")

def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def _prepare_md_bytes(md: str, add_bom: bool = False) -> bytes:
    """Normaliza e prepara os bytes do Markdown para upload (UTF-8)."""
    md = unicodedata.normalize("NFC", md)
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    md = md.replace("\ufeff", "")  # remove BOM existente
    md = "".join(c for c in md if c == "\n" or c == "\t" or ord(c) >= 32)
    raw = md.encode("utf-8", "strict")
    return (b"\xef\xbb\xbf" + raw) if add_bom else raw


def verify_public_content(public_url: str, original_bytes: bytes, allow_plain: bool = False) -> tuple[bool, bool, str]:
    """
    Valida round-trip e o Content-Type.
    Retorna (ok, needs_plain_fallback, content_type_str).
    """
    r = requests.get(public_url, timeout=20)
    r.raise_for_status()
    ctype = (r.headers.get("Content-Type") or "").lower()

    # Aceita text/markdown e (opcionalmente) text/plain, mas exige charset=utf-8
    type_ok = ("text/markdown" in ctype) or (allow_plain and "text/plain" in ctype)
    utf8_ok = "charset=utf-8" in ctype
    same = _sha256(r.content) == _sha256(original_bytes)

    if not type_ok or not utf8_ok:
        return (same and utf8_ok and type_ok), True, ctype  # sugere fallback p/ text/plain
    if not same:
        raise RuntimeError("Arquivo no Storage difere do original (hash mismatch)")
    return True, False, ctype


# =========================================================
# Imagem: hashing e helpers
# =========================================================

def _flatten_rgba_to_rgb(im: Image.Image, bg=(255, 255, 255)) -> Image.Image:
    if im.mode in ("RGBA", "LA"):
        bg_img = Image.new("RGBA", im.size, bg + (255,))
        return Image.alpha_composite(bg_img, im.convert("RGBA")).convert("RGB")
    return im.convert("RGB")

def _smart_trim(im: Image.Image, bg_thresh: int = 245, pad: int = 2) -> Image.Image:
    im = _flatten_rgba_to_rgb(im).convert("RGB")
    arr = np.array(im)
    mask = ~(
        (arr[..., 0] >= bg_thresh)
        & (arr[..., 1] >= bg_thresh)
        & (arr[..., 2] >= bg_thresh)
    )
    ys, xs = np.where(mask)
    if ys.size == 0 or xs.size == 0:
        return im
    y1, y2 = max(ys.min() - pad, 0), min(ys.max() + pad, arr.shape[0] - 1)
    x1, x2 = max(xs.min() - pad, 0), min(xs.max() + pad, arr.shape[1] - 1)
    return im.crop((x1, y1, x2 + 1, y2 + 1))

def _ahash_from_image(img: Image.Image, hash_size: int = 8) -> int:
    g = img.convert("L").resize((hash_size, hash_size), Image.BILINEAR)
    px = list(g.getdata())
    avg = sum(px) / len(px)
    bits = 0
    for i, p in enumerate(px):
        if p > avg:
            bits |= (1 << i)
    return bits

def _dhash_from_image(img: Image.Image, hash_size: int = 8) -> int:
    g = _flatten_rgba_to_rgb(img).convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
    pixels = list(g.getdata())
    bits = 0
    for row in range(hash_size):
        for col in range(hash_size):
            left = pixels[row * (hash_size + 1) + col]
            right = pixels[row * (hash_size + 1) + col + 1]
            if right > left:
                bits |= (1 << (row * hash_size + col))
    return bits

def _phash_from_image(img: Image.Image, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    img = _flatten_rgba_to_rgb(img).convert("L").resize(
        (hash_size * highfreq_factor, hash_size * highfreq_factor), Image.BILINEAR
    )
    arr = np.asarray(img, dtype=np.float32)
    dct = cv2.dct(arr)
    dctlow = dct[:hash_size, :hash_size]
    med = np.median(dctlow)
    bits = (dctlow > med).astype(np.uint8).flatten()
    out = 0
    for i, b in enumerate(bits):
        if b:
            out |= (1 << i)
    return out

def _hist_signature(img: Image.Image, bins: int = 32) -> np.ndarray:
    rgb = np.array(_flatten_rgba_to_rgb(img))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sig = []
    for ch in range(3):
        h = cv2.calcHist([hsv], [ch], None, [bins], [0, 256]).astype("float32")
        h = h / (h.sum() + 1e-9)
        sig.append(h)
    return np.vstack(sig).reshape(-1)

def _bhattacharyya(h1: np.ndarray, h2: np.ndarray) -> float:
    h1 = h1.reshape(-1).astype("float32")
    h2 = h2.reshape(-1).astype("float32")
    return float(np.sqrt(1.0 - np.sum(np.sqrt(h1 * h2 + 1e-12))))

def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def _parse_multi_list(values: List[str]) -> List[str]:
    out: List[str] = []
    for v in values or []:
        for piece in str(v).split(","):
            p = piece.strip()
            if p:
                out.append(p)
    return out

def _aspect_ratio(w: int | None, h: int | None) -> float:
    return (float(w) / float(h)) if w and h and h != 0 else 0.0

def _has_brazil_flag_colors_pil(img: Image.Image) -> bool:
    rgb = np.array(_flatten_rgba_to_rgb(img))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    H, S = hsv[..., 0], hsv[..., 1]
    sat = S > 60
    yellow = ((H >= 20) & (H <= 35) & sat)
    green = ((H >= 40) & (H <= 85) & sat)
    blue = ((H >= 90) & (H <= 130) & sat)
    total = hsv.shape[0] * hsv.shape[1]
    if total == 0:
        return False
    py = np.count_nonzero(yellow) / total
    pg = np.count_nonzero(green) / total
    pb = np.count_nonzero(blue) / total
    return (py > 0.05 and pg > 0.08 and pb > 0.05)

# =========================================================
# Regex globais para imagens em Markdown/HTML
# =========================================================

IMG_MD_PATTERN = re.compile(r"!\[[^\]]*\]\((?!https?://)([^)]+)\)")
IMG_HTML_PATTERN = re.compile(r'<img[^>]+src=["\'](?!https?://)([^"\']+)["\'][^>]*>', re.I)

# =========================================================
# Remoção de imagens específicas (por template/regex/área)
# =========================================================

def remove_specific_images_from_md(
    md: str,
    base_dir: Path,
    template_paths: List[str] | None = None,
    hash_threshold: int = 6,                 # aHash
    name_regex: str | None = None,
    max_area: int | None = None,
    tpl_phash_threshold: int = 22,           # pHash
    aspect_tol: float = 0.20,                # ±20%
    color_check: bool = False,
    debug: bool = False,
) -> tuple[str, List[Path]]:

    tpl_ah, tpl_ph, tpl_dh, tpl_aspects = [], [], [], []

    for t in _parse_multi_list(template_paths or []):
        tpath = (Path(t).expanduser().resolve())
        try:
            with Image.open(tpath) as tim:
                tim = _smart_trim(tim)
                tpl_ah.append(_ahash_from_image(tim))
                tpl_ph.append(_phash_from_image(tim))
                tpl_dh.append(_dhash_from_image(tim))
                tw, th = tim.size
                tpl_aspects.append(_aspect_ratio(tw, th))
        except Exception as e:
            if debug:
                LOGGER.debug("Falha lendo template %s: %s", tpath.name, e)

    name_pat = re.compile(name_regex, re.IGNORECASE) if name_regex else None
    removed: List[Path] = []

    def _should_drop(img_path: Path) -> bool:
        if not img_path.exists():
            return False
        try:
            fname = img_path.name
            if name_pat and name_pat.search(fname):
                if debug:
                    LOGGER.debug("Nome bateu regex: %s", fname)
                return True

            with Image.open(img_path) as im:
                im = _smart_trim(im)
                im = _flatten_rgba_to_rgb(im)
                w, h = im.size
                area = (w * h) if (w and h) else None

                # Regra 1: tamanho
                if (max_area is not None) and (area is not None) and (area <= max_area):
                    if debug:
                        LOGGER.debug("Área <= %s: %s (%sx%s=%s)", max_area, fname, w, h, area)
                    return True

                # Regra 2: template (aHash/pHash/dHash + aspecto + (opcional) cores)
                ah = _ahash_from_image(im)
                ph = _phash_from_image(im)
                dh = _dhash_from_image(im)
                asp = _aspect_ratio(w, h)

                min_da = min((_hamming(ah, th) for th in tpl_ah), default=None)
                min_dp = min((_hamming(ph, th) for th in tpl_ph), default=None)
                min_dd = min((_hamming(dh, th) for th in tpl_dh), default=None)

                ph_ok = (min_dp is not None and min_dp <= tpl_phash_threshold)
                ah_ok = (min_da is not None and min_da <= hash_threshold)
                dh_ok = (min_dd is not None and min_dd <= max(hash_threshold, 8))

                min_dasp = None
                if tpl_aspects and asp > 0:
                    min_dasp = min(
                        (abs(asp - ta) / max(ta, asp, 1e-6) for ta in tpl_aspects),
                        default=None,
                    )

                hash_hit = ph_ok and (ah_ok or dh_ok)
                aspect_hit = (min_dasp is None) or (min_dasp <= aspect_tol)
                color_hit = (not color_check) or _has_brazil_flag_colors_pil(im)

                if debug and (min_da is not None or min_dp is not None or min_dd is not None):
                    LOGGER.debug(
                        "drop-check %s %sx%s aHash=%s pHash=%s dHash=%s Δaspect≈%s%% coresOK=%s",
                        fname, w, h, min_da, min_dp, min_dd,
                        f"{100*(min_dasp or 0):.1f}", color_hit
                    )

                if hash_hit and aspect_hit and color_hit:
                    return True

        except Exception as e:
            if debug:
                LOGGER.debug("Erro lendo %s: %s", img_path.name, e)
            return False
        return False

    def _replacer_md(m: re.Match) -> str:
        rel = m.group(1)
        rel_clean = rel.strip().lstrip("./").replace("\\", "/")
        abs_path = (base_dir / rel_clean).resolve()
        if _should_drop(abs_path):
            removed.append(abs_path)
            return ""
        return m.group(0)

    def _replacer_html(m: re.Match) -> str:
        rel = m.group(1)
        rel_clean = rel.strip().lstrip("./").replace("\\", "/")
        abs_path = (base_dir / rel_clean).resolve()
        if _should_drop(abs_path):
            removed.append(abs_path)
            return ""
        return m.group(0)

    new_md = IMG_MD_PATTERN.sub(_replacer_md, md)
    new_md = IMG_HTML_PATTERN.sub(_replacer_html, new_md)
    new_md = re.sub(r"\n{3,}", "\n\n", new_md).strip()
    return new_md, removed

# =========================================================
# Remoção de imagens repetidas (dedup por conteúdo)
# =========================================================

def drop_repeated_images_in_md(
    md: str,
    base_dir: Path,
    hash_threshold: int = 10,           # usado apenas em mode="loose"
    min_count: int = 3,
    size_tol: float = 0.10,
    keep: str = "none",
    area_min: int | None = None,
    area_max: int | None = None,
    debug: bool = False,
    *,
    mode: str = "strict",               # "strict" | "loose"
    phash_threshold: int = 8,
    dhash_threshold: int = 8,
    hist_bhatta_max: float = 0.08,
    consensus: int = 3,
    size_tol_strict: float | None = 0.05,
) -> tuple[str, dict]:

    matches = list(IMG_MD_PATTERN.finditer(md))
    if not matches:
        return md, {}

    info = []
    for m in matches:
        rel = m.group(1).strip()
        rel_clean = rel.lstrip("./").replace("\\", "/")
        path = (base_dir / rel_clean).resolve()

        data = {
            "span": m.span(),
            "rel": rel,
            "path": path,
            "w": None,
            "h": None,
            "area": None,
            "ah": None,
            "ph": None,
            "dh": None,
            "hist": None,
            "valid": False,
        }

        if path.exists():
            try:
                with Image.open(path) as im:
                    im2 = _flatten_rgba_to_rgb(im)
                    w, h = im2.size
                    data["w"], data["h"] = w, h
                    area = w * h if (w and h) else None
                    data["area"] = area

                    if area is not None:
                        if (area_min is not None and area < area_min) or (
                            area_max is not None and area > area_max
                        ):
                            data["valid"] = False
                        else:
                            data["ah"] = _ahash_from_image(im2)
                            if mode == "strict":
                                data["ph"] = _phash_from_image(im2)
                                data["dh"] = _dhash_from_image(im2)
                                data["hist"] = _hist_signature(im2)
                            data["valid"] = True
            except Exception:
                data["valid"] = False

        info.append(data)

    def _size_ok(ai, aj) -> bool:
        wi, hi, wj, hj = ai["w"], ai["h"], aj["w"], aj["h"]
        if not all([wi, hi, wj, hj]):
            return False
        tol = size_tol_strict if (mode == "strict" and size_tol_strict is not None) else size_tol
        return (abs(wi - wj) / max(wi, wj) <= tol) and (abs(hi - hj) / max(hi, hj) <= tol)

    def similar(i: int, j: int) -> bool:
        ai, aj = info[i], info[j]
        if not (ai["valid"] and aj["valid"]):
            return False
        if not _size_ok(ai, aj):
            return False

        if mode == "loose":
            return _hamming(int(ai["ah"]), int(aj["ah"])) <= hash_threshold

        votes = 0
        ah = _hamming(int(ai["ah"]), int(aj["ah"]))
        if ah <= min(hash_threshold, 8):
            votes += 1

        if ai["ph"] is not None and aj["ph"] is not None:
            ph = _hamming(int(ai["ph"]), int(aj["ph"]))
            if ph <= phash_threshold:
                votes += 1

        if ai["dh"] is not None and aj["dh"] is not None:
            dh = _hamming(int(ai["dh"]), int(aj["dh"]))
            if dh <= dhash_threshold:
                votes += 1

        if ai["hist"] is not None and aj["hist"] is not None:
            hb = _bhattacharyya(ai["hist"], aj["hist"])
            if hb <= hist_bhatta_max:
                votes += 1

        if debug:
            LOGGER.debug(
                "[repeated] sizes=%sx%s~%sx%s ah=%s ph=%s dh=%s hb≈%s votes=%s/4",
                ai["w"], ai["h"], aj["w"], aj["h"],
                ah,
                (_hamming(int(ai["ph"]), int(aj["ph"])) if (ai["ph"] is not None and aj["ph"] is not None) else "-"),
                (_hamming(int(ai["dh"]), int(aj["dh"])) if (ai["dh"] is not None and aj["dh"] is not None) else "-"),
                (_bhattacharyya(ai["hist"], aj["hist"]) if (ai["hist"] is not None and aj["hist"] is not None) else "-"),
                votes,
            )
        return votes >= consensus

    clusters: list[dict] = []
    for i in range(len(info)):
        if not info[i]["valid"]:
            continue
        placed = False
        for c in clusters:
            rep = c["rep"]
            if similar(i, rep):
                c["members"].append(i)
                placed = True
                break
        if not placed:
            clusters.append({"rep": i, "members": [i]})

    to_drop: set[int] = set()
    report: Dict[int, Dict[str, Any]] = {}
    cid = 1
    for c in clusters:
        members = sorted(c["members"])
        if len(members) >= min_count:
            r = info[c["rep"]]
            report[cid] = {
                "count": len(members),
                "sample": str(r["path"]) if r["path"] else "",
                "w": r["w"],
                "h": r["h"],
            }
            cid += 1
            if keep == "first":
                for j, k in enumerate(members):
                    if j > 0:
                        to_drop.add(k)
            else:
                to_drop.update(members)

    out = []
    last = 0
    for i, m in enumerate(matches):
        s, e = m.span()
        out.append(md[last:s])
        if i not in to_drop:
            out.append(md[s:e])
        last = e
    out.append(md[last:])
    new_md = "".join(out)
    new_md = re.sub(r"\n{3,}", "\n\n", new_md).strip()
    return new_md, report

# =========================================================
# Watermark cleaner (remove.py)
# =========================================================

def run_watermark_cleaner(
    in_pdf: Path,
    out_pdf: Path,
    runner: str = "native",           # "native" | "subprocess"
    wm_script: Path | None = None,    # usado no subprocess
    py_exec: str | None = None,       # usado no subprocess
    dpi: int = 220,
    inpaint: str = "telea",           # "telea" | "ns"
    use_pdfcpu: bool = True,
    mask_debug: Path | None = None,
) -> bool:
    try:
        if runner == "native":
            try:
                import importlib.util
                rm_path = (Path(__file__).parent / "remove.py").resolve()
                spec = importlib.util.spec_from_file_location("pdfwm_remove", str(rm_path))
                if spec is None or spec.loader is None:
                    raise ImportError("spec loader vazio")
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore
                report = mod.process_pdf(
                    in_pdf=in_pdf,
                    out_pdf=out_pdf,
                    dpi=dpi,
                    inpaint_method=inpaint,
                    use_pdfcpu=use_pdfcpu,
                    mask_debug_dir=mask_debug,
                )
                LOGGER.info("[wm] Concluído (native). Relatório (500 chars): %s",
                            json.dumps(report, ensure_ascii=False)[:500])
                return out_pdf.exists() and out_pdf.stat().st_size > 0
            except Exception as e:
                LOGGER.debug("[wm] native falhou (%s). Tentando subprocess…", e)
                runner = "subprocess"

        if runner == "subprocess":
            wm_script = Path("remove.py").resolve() if wm_script is None else wm_script
            py_exec = sys.executable if py_exec is None else py_exec
            cmd = [
                str(py_exec), str(wm_script),
                "--input", str(in_pdf),
                "--output", str(out_pdf),
                "--dpi", str(dpi),
                "--inpaint", inpaint,
                "--pdfcpu", "true" if use_pdfcpu else "false",
            ]
            if mask_debug:
                cmd += ["--mask-debug", str(mask_debug)]
            LOGGER.info("[wm] Executando: %s", " ".join(cmd))
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if r.stdout:
                LOGGER.debug("[wm] stdout: %s", r.stdout[:1000])
            if r.stderr:
                LOGGER.debug("[wm] stderr: %s", r.stderr[:1000])
            return out_pdf.exists() and out_pdf.stat().st_size > 0

        return False
    except Exception as e:
        LOGGER.error("[wm] Erro ao rodar cleaner: %s", e)
        return False

# =========================================================
# Download do PDF
# =========================================================

def download_pdf(pdf_url: str, dest_path: Path, timeout: int = 30) -> None:
    resp = requests.get(pdf_url, stream=True, timeout=timeout)
    resp.raise_for_status()
    ctype = resp.headers.get("Content-Type", "")
    if "pdf" not in ctype.lower() and not pdf_url.lower().endswith(".pdf"):
        LOGGER.warning("Content-Type não parece PDF (%s). Continuando…", ctype)
    with open(dest_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)

# =========================================================
# Conversão Docling → Markdown (referenciando imagens)
# =========================================================

def convert_with_docling(
    pdf_path: Path,
    out_dir: Path,
    images_scale: float = 2.0,
    include_page_images: bool = False,
) -> Tuple[str, Path]:
    ensure_dir(out_dir)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = images_scale
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_page_images = bool(include_page_images)

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    conv_res = converter.convert(str(pdf_path))

    md_tmp = out_dir / "_docling.md"
    from docling_core.types.doc import ImageRefMode
    conv_res.document.save_as_markdown(md_tmp, image_mode=ImageRefMode.REFERENCED)

    md_text = md_tmp.read_text(encoding="utf-8")
    if not md_text.strip():
        raise RuntimeError("Docling retornou Markdown vazio.")
    return md_text, md_tmp

def convert_fallback_pymupdf4llm(pdf_path: Path) -> str:
    if not HAS_PYMUPDF4LLM:
        return ""
    try:
        return pymupdf4llm.to_markdown(str(pdf_path)) or ""
    except Exception:
        return ""

# =========================================================
# Sanitização de links/e-mails
# =========================================================

def sanitize_markdown(
    md: str,
    allowed_domains: List[str] | None = None,
    remove_emails: bool = True,
) -> str:
    allowed = [d.lower().strip() for d in (allowed_domains or []) if d.strip()]

    def _is_allowed(url: str) -> bool:
        if not url:
            return False
        u = url.strip().lower()
        if u.startswith("www."):
            u = "http://" + u
        elif "://" not in u:
            u = "http://" + u
        try:
            host = urlparse(u).netloc
            return any(host.endswith(dom) for dom in allowed)
        except Exception:
            return False

    def _repl_inline(m: re.Match) -> str:
        text, url = m.group("text"), m.group("url")
        return f"[{text}]({url})" if _is_allowed(url) else text

    md = re.sub(r"(?<!!)\[(?P<text>[^\]]+)\]\((?P<url>[^)]+)\)", _repl_inline, md, flags=re.IGNORECASE)

    def _repl_autolink(m: re.Match) -> str:
        url = m.group("url")
        return f"<{url}>" if _is_allowed(url) else ""

    md = re.sub(r"<(?P<url>https?://[^>\s]+|www\.[^>\s]+|[a-z0-9][a-z0-9\-\.]+\.[a-z]{2,24}[^\s>]*)>", _repl_autolink, md, flags=re.IGNORECASE)

    def _repl_http(m: re.Match) -> str:
        url = m.group("url")
        return url if _is_allowed(url) else ""

    md = re.sub(r"(?<!\()(?<!\[)(?P<url>https?://[^\s<>\)]+)", _repl_http, md, flags=re.IGNORECASE)

    def _repl_bare(m: re.Match) -> str:
        url = m.group("url")
        return url if _is_allowed(url) else ""

    md = re.sub(r"(?:(?<=^)|(?<=[\s\(\[{,;]))(?P<url>(?:www\.)[^\s<>\)]{3,})", _repl_bare, md, flags=re.IGNORECASE | re.MULTILINE)
    md = re.sub(
        r"(?:(?<=^)|(?<=[\s\(\[{,;]))(?P<url>(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}(?:/[^\s<>\)]*)?)",
        _repl_bare,
        md,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    def _repl_ref(m: re.Match) -> str:
        label, url = m.group("label"), m.group("url")
        return m.group(0) if _is_allowed(url) else f"[{label}]:"

    md = re.sub(r"^\[(?P<label>[^\]]+)\]:\s*(?P<url>\S+)\s*$", _repl_ref, md, flags=re.IGNORECASE | re.MULTILINE)

    if remove_emails:
        md = re.sub(r"(?<!\w)[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24}(?!\w)", "", md, flags=re.IGNORECASE)

    md = re.sub(r"[ \t]{2,}", " ", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()

# =========================================================
# Polimento de Markdown (headings/quebras/imagens)
# =========================================================

def polish_markdown(md: str) -> str:
    drop_patterns = [
        r"onlinedoctranslator",
        r"https?://\S*?cambioautomaticodobrasil\.com\.br\S*",
        r"^\s*Página\s+\d+\s*/\s*\d+\s*$",
    ]
    keep_lines = []
    for line in md.splitlines():
        if any(re.search(p, line, re.IGNORECASE) for p in drop_patterns):
            continue
        keep_lines.append(line)
    md = "\n".join(keep_lines)

    def _fix_img_path(m: re.Match) -> str:
        alt = m.group(1)
        path = m.group(2).replace("\\", "/")
        return f"![{alt}]({path})"

    md = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _fix_img_path, md)

    md = re.sub(r"(?m)^##\s*(.+?)\s*3\s*\n+(?:^##\s*)?ª geração:?\s*$", r"## \1 3ª geração", md)
    md = re.sub(r"(?m)^(## .+)\n+\1(\n+|$)", r"\1\2", md)
    md = re.sub(r"(?m)^(P[0-9A-Z]{4}\b[^\n]*)", r"- \1", md)
    md = md.replace("ÿ", "Ω")
    md = re.sub(r"[ \t]{2,}", " ", md)
    md = re.sub(r"(\w+)-\n(\w+)", r"\1\2", md)

    joined_lines, buffer = [], ""

    def _is_block_boundary(s: str) -> bool:
        s = s.strip()
        return (
            not s
            or s.startswith("#")
            or s.startswith("- ")
            or s.startswith("* ")
            or s.startswith("> ")
            or s.startswith("!")
            or s.startswith("|")
            or s.startswith("```")
            or re.match(r"^\d+\.\s", s) is not None
        )

    def _ends_sentence(s: str) -> bool:
        return bool(re.search(r"[.!?:;)]\s*$", s.strip()))

    for line in md.splitlines():
        if _is_block_boundary(line):
            if buffer:
                joined_lines.append(buffer.strip())
                buffer = ""
            joined_lines.append(line)
        else:
            if not buffer:
                buffer = line.strip()
            else:
                sep = "" if buffer.endswith("-") else " "
                if _ends_sentence(buffer):
                    joined_lines.append(buffer)
                    buffer = line.strip()
                else:
                    buffer += sep + line.strip()
    if buffer:
        joined_lines.append(buffer.strip())
    md = "\n".join(joined_lines)

    lines = md.splitlines()
    out = []
    current_heading = None
    fig_counter = 0
    img_pat = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    for line in lines:
        if line.strip().startswith("## "):
            current_heading = re.sub(r"^##\s*", "", line).strip()

        def _alt_enrich(m: re.Match) -> str:
            nonlocal fig_counter, current_heading
            alt, path = (m.group(1) or "").strip(), m.group(2).strip()
            if not alt or alt.lower() in {"image", "imagem"}:
                fig_counter += 1
                context = f" — {current_heading}" if current_heading else ""
                alt = f"Figura {fig_counter}{context}"
            return f"![{alt}]({path})"

        line = img_pat.sub(_alt_enrich, line)
        out.append(line)

    md = "\n".join(out)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()

def _normalize_bullets_and_lists(md: str) -> str:
    lines = md.splitlines()
    out = []
    bullet_pat = re.compile(r"^\s*[•●○▪▫◦·–—]\s+")
    num_pat = re.compile(r"^\s*(\d+|[a-zA-Z]|[ivxlcdmIVXLCDM]+)[\.\)]\s+")

    for ln in lines:
        if bullet_pat.match(ln):
            ln = bullet_pat.sub("- ", ln)
        m = num_pat.match(ln)
        if m:
            token = m.group(1)
            ln = num_pat.sub(f"{token}. ", ln, count=1)
        out.append(ln)
    return "\n".join(out)

def _fix_tables(md: str) -> str:
    lines = md.splitlines()
    out, block = [], []

    def is_table_line(s: str) -> bool:
        return s.strip().count("|") >= 2 and s.strip().startswith("|") and s.strip().endswith("|")

    i = 0
    while i < len(lines):
        if is_table_line(lines[i]):
            block = [lines[i]]
            i += 1
            while i < len(lines) and is_table_line(lines[i]):
                block.append(lines[i])
                i += 1
            if len(block) >= 2:
                cols = block[0].strip().strip("|").split("|")
                if not re.match(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", block[1]):
                    sep = "|" + "|".join([" --- " for _ in cols]) + "|"
                    block.insert(1, sep)
            out.extend(block)
        else:
            out.append(lines[i])
            i += 1

    return "\n".join(out)

def _drop_repeated_short_lines(md: str, min_count: int = 6, max_len: int = 120) -> str:
    freq: Dict[str, int] = {}
    lines = md.splitlines()
    for ln in lines:
        s = ln.strip()
        if not s or len(s) > max_len:
            continue
        if s.startswith(("#", "-", "|", "!", ">")):
            continue
        freq[s] = freq.get(s, 0) + 1

    bad = {s for s, c in freq.items() if c >= min_count}

    out = []
    for ln in lines:
        s = ln.strip()
        if s in bad:
            continue
        out.append(ln)
    return "\n".join(out)

def polish_markdown_fidelity(md: str, drop_repeated_lines: bool = False,
                             repeated_min_count: int = 6, repeated_max_len: int = 120) -> str:
    md = _normalize_bullets_and_lists(md)
    md = _fix_tables(md)
    if drop_repeated_lines:
        md = _drop_repeated_short_lines(md, min_count=repeated_min_count, max_len=repeated_max_len)
    return md

def _normkey(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s).strip().lower()

def _normalize_title_variants(s: str) -> str:
    s = re.sub(r"\bgen\s*([23])\b", r"\1ª geração", s, flags=re.IGNORECASE)
    s = re.sub(r"\bsolenoide\b", "solenóide", s, flags=re.IGNORECASE)
    s = re.sub(
        r"(?i)identificação do sol[eé]n[oó]ide\s+09g\s+(2|3)ª geração",
        lambda m: f"Identificação do solenóide 09G {m.group(1)}ª geração",
        s,
    )
    return s

def _dedupe_headings_window(md: str, window: int = 12) -> str:
    lines = md.splitlines()
    out = []
    recent: list[tuple[int, str]] = []
    for ln in lines:
        if ln.strip().startswith("## "):
            title = ln.strip()[3:].strip()
            title = _normalize_title_variants(title)
            ln_fixed = "## " + title
            key = _normkey(title)
            already = any(k == key for _, k in recent[-window:])
            if not already:
                out.append(ln_fixed)
                recent.append((len(out) - 1, key))
        else:
            out.append(ln)
            if len(recent) > 200:
                recent = recent[-200:]
    return "\n".join(out)

def _drop_repeated_subtitles(md: str) -> str:
    target = _normkey("Função, Resistência, Possível Código DTC.")
    lines = md.splitlines()
    out = []
    after_heading_seen = False
    used_once_for_key: set[str] = set()
    last_title_key = None

    for ln in lines:
        if ln.strip().startswith("## "):
            out.append(ln)
            last_title_key = _normkey(ln.strip()[3:])
            after_heading_seen = True
            continue

        if after_heading_seen and last_title_key:
            if _normkey(ln) == target:
                if last_title_key not in used_once_for_key:
                    out.append(ln)
                    used_once_for_key.add(last_title_key)
                after_heading_seen = False
                continue

        out.append(ln)
        if ln.strip():
            after_heading_seen = False
    return "\n".join(out)

def _normalize_block_spacing(md: str) -> str:
    md = re.sub(r"\n{3,}", "\n\n", md.strip())
    lines = md.splitlines()
    out = []

    for ln in lines:
        is_heading = ln.strip().startswith("## ")
        is_image = bool(re.match(r"!\[[^\]]*\]\([^)]+\)", ln.strip()))
        if is_heading:
            if out and out[-1].strip() != "":
                out.append("")
            out.append(ln)
        elif is_image:
            if out and out[-1].strip() != "":
                out.append("")
            out.append(ln)
            out.append("")
        else:
            out.append(ln)

    s = "\n".join(out)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"(## [^\n]+)\n(?!\n)", r"\1\n\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _fix_double_slash_urls(md: str) -> str:
    def repl(m: re.Match) -> str:
        alt, url = m.group(1), m.group(2)
        safe = url.replace("https://", "https:__").replace("http://", "http:__")
        safe = re.sub(r"/{2,}", "/", safe)
        safe = safe.replace("https:__", "https://").replace("http:__", "http://")
        return f"![{alt}]({safe})"
    return re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", repl, md)

def polish_markdown_fidelity_plus(md: str) -> str:
    md = _dedupe_headings_window(md, window=12)
    md = _drop_repeated_subtitles(md)
    md = _explode_dtc_paragraphs(md)
    md = _normalize_block_spacing(md)
    md = _fix_double_slash_urls(md)
    return md

def _explode_dtc_paragraphs(md: str) -> str:
    def split_dtc_block(txt: str) -> list[str]:
        parts = re.findall(r"(P\d{4}\b.*?)(?=P\d{4}\b|$)", txt)
        return [p.strip() for p in parts if p.strip()]

    lines = md.splitlines()
    out = []
    for ln in lines:
        has_multi = re.search(r"(?:^|[^A-Z0-9])P\d{4}\b.*P\d{4}\b", ln)
        if has_multi:
            chunks = split_dtc_block(ln)
            for c in chunks:
                out.append(c if c.startswith("- ") else "- " + c)
        else:
            if re.search(r"(?:^|[^A-Z0-9])P\d{4}\b", ln) and not ln.strip().startswith("- "):
                out.append("- " + ln.strip())
            else:
                out.append(ln)
    return "\n".join(out)

# =========================================================
# Remoção de rodapés/branding corporativo
# =========================================================

def drop_corporate_branding(
    md: str,
    brand_terms: list[str] | None = None,
    max_len: int = 140,
) -> str:
    brand_terms = [t for t in (brand_terms or []) if t.strip()]
    corp_core = r"(?:ltda\.?|s\.?/?a\.?|eireli|mei\b|me\b|epp\b|holding|ind[uú]stria|com[eé]rcio|servi[cç]os|inc\.?|llc)"
    contact_core = r"(?:cnpj|ie\b|im\b|telefone|tel\.|whatsapp|email|e-mail|site|www\.)"
    cnpj_pat = r"\b\d{2}\.?\d{3}\.?\d{3}/?\d{4}-?\d{2}\b"
    custom = "|".join(re.escape(t) for t in brand_terms) if brand_terms else ""
    token_pat = re.compile(rf"(?i)\b(?:{corp_core}|{contact_core}{'|' + custom if custom else ''})\b|{cnpj_pat}")

    out = []
    for ln in md.splitlines():
        s = ln.strip()
        if not s:
            out.append(ln)
            continue
        if s.startswith(("## ", "- ", "* ", ">", "!", "|", "```")):
            out.append(ln)
            continue
        if len(s) <= max_len and token_pat.search(s):
            continue
        out.append(ln)
    return "\n".join(out)

# =========================================================
# Coleta e reescrita de imagens no Markdown
# =========================================================

def collect_local_images_from_md(md: str, base_dir: Path) -> List[Path]:
    matches = IMG_MD_PATTERN.findall(md)
    images: List[Path] = []
    seen = set()
    for rel in matches:
        rel_clean = rel.strip().lstrip("./")
        img_path = (base_dir / rel_clean).resolve()
        if img_path.exists() and img_path.is_file() and img_path not in seen:
            images.append(img_path)
            seen.add(img_path)
    return images

def rewrite_image_paths_in_md(md: str, base_dir: Path, url_map: Dict[Path, str]) -> str:
    def _replace(match: re.Match) -> str:
        rel = match.group(1)
        rel_clean = rel.strip().lstrip("./")
        abs_path = (base_dir / rel_clean).resolve()
        new_url = url_map.get(abs_path)
        if not new_url:
            return match.group(0)
        full = match.group(0)
        return full.replace(f"({rel})", f"({new_url})")
    return IMG_MD_PATTERN.sub(_replace, md)

# =========================================================
# Supabase: cliente e uploads
# =========================================================

def supabase_client_from_env():
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise RuntimeError("SUPABASE_URL ou SUPABASE_KEY ausentes no .env")
    return create_client(supabase_url, supabase_key)

def upload_text_to_supabase(
    client,
    bucket: str,
    remote_path: str,
    text: str,
    content_type: str = "text/markdown; charset=utf-8",
    upsert: bool = True,
    add_bom: bool = False,   # <— NOVO
) -> str:
    storage = client.storage
    data = _prepare_md_bytes(text, add_bom=add_bom)

    tried_opts = [
        {"contentType": content_type, "upsert": "true", "cacheControl": "public, max-age=31536000, immutable"},
        {"content_type": content_type, "upsert": "true", "cacheControl": "public, max-age=31536000, immutable"},
        # fallbacks
        {"contentType": "text/plain; charset=utf-8", "upsert": "true"},
        {"content_type": "text/plain; charset=utf-8", "upsert": "true"},
    ]
    last_err = None
    for fo in tried_opts:
        try:
            storage.from_(bucket).upload(remote_path, data, file_options=fo)
            break
        except Exception as e:
            last_err = e
    else:
        raise RuntimeError(f"Falha no upload do texto para {remote_path}: {last_err}")

    public = storage.from_(bucket).get_public_url(remote_path)
    public_url = public.get("publicUrl") if isinstance(public, dict) else str(public)
    if not public_url:
        raise RuntimeError(f"Não foi possível obter URL pública para {remote_path}")

    ok, needs_plain, ctype = verify_public_content(public_url, data, allow_plain=True)
    if not ok and needs_plain:
        # reenvia forçando text/plain; charset=utf-8 (alguns CDNs tratam melhor)
        storage.from_(bucket).upload(
            remote_path, data, file_options={"contentType": "text/plain; charset=utf-8", "upsert": "true"}
        )
        ok2, _, ctype2 = verify_public_content(public_url, data, allow_plain=False)
        if not ok2:
            raise RuntimeError(f"Public URL não está servindo UTF-8 corretamente (Content-Type atual: {ctype2})")

    return public_url

def upload_images_to_supabase(
    client,
    bucket: str,
    pdf_id: str,
    local_images: List[Path],
    storage_prefix: str = "images",
) -> Dict[Path, str]:
    mapping: Dict[Path, str] = {}
    storage = client.storage

    FileOptions = None
    try:
        from supabase.storage.types import FileOptions as _FileOptions  # type: ignore
        FileOptions = _FileOptions
    except Exception:
        pass

    storage_prefix = (storage_prefix or "").strip().strip("/")

    for img_path in local_images:
        remote_name = slug_filename(img_path.name)
        remote_path = f"{storage_prefix}/{pdf_id}/{remote_name}" if storage_prefix else f"{pdf_id}/{remote_name}"

        ext = img_path.suffix.lower()
        if ext in {".png"}:
            ctype = "image/png"
        elif ext in {".jpg", ".jpeg"}:
            ctype = "image/jpeg"
        elif ext in {".webp"}:
            ctype = "image/webp"
        else:
            ctype = "application/octet-stream"

        data = img_path.read_bytes()
        uploaded = False
        last_err = None

        if FileOptions is not None:
            try:
                opts = FileOptions(content_type=ctype, upsert=True)
                storage.from_(bucket).upload(remote_path, data, file_options=opts)
                uploaded = True
            except Exception as e:
                last_err = e

        if not uploaded:
            try:
                storage.from_(bucket).upload(
                    remote_path, data, file_options={"content_type": ctype, "upsert": "true"}
                )
                uploaded = True
            except Exception as e:
                last_err = e

        if not uploaded:
            try:
                storage.from_(bucket).upload(
                    remote_path, data, file_options={"contentType": ctype, "upsert": "true"}
                )
                uploaded = True
            except Exception as e:
                last_err = e

        if not uploaded:
            raise RuntimeError(f"Falha no upload de {img_path.name}: {last_err}")

        public = storage.from_(bucket).get_public_url(remote_path)
        public_url = public.get("publicUrl") if isinstance(public, dict) else str(public)
        if not public_url:
            raise RuntimeError(f"Não foi possível obter URL pública para {remote_path}")

        mapping[img_path] = public_url

    return mapping

# =========================================================
# Pipeline principal (single PDF) e modo batch (DB)
# =========================================================

def pipeline_process_pdf(pdf_id: str, pdf_url: str, args) -> tuple[str, Path]:
    allowed_domains = [d.strip() for d in (args.allowed_domains or "").split(",") if d.strip()]
    pdf_id = slug_filename(str(pdf_id))

    work_dir = Path(args.out_dir) / pdf_id
    img_dir = work_dir / "images"
    ensure_dir(work_dir)
    ensure_dir(img_dir)

    local_pdf = work_dir / f"{pdf_id}.pdf"

    LOGGER.info("[1/6] Baixando PDF… %s", pdf_url)
    download_pdf(pdf_url, local_pdf)

    if args.wm_clean:
        LOGGER.info("[2a] Limpando marcas d’água (pikepdf/pdfcpu/inpaint)…")
        cleaned_pdf = work_dir / f"{pdf_id}.wmclean.pdf"
        mask_dir = Path(args.wm_mask_debug).resolve() if args.wm_mask_debug else None
        ok = run_watermark_cleaner(
            in_pdf=local_pdf,
            out_pdf=cleaned_pdf,
            runner=args.wm_runner,
            wm_script=Path(args.wm_script).resolve(),
            py_exec=args.wm_python,
            dpi=args.wm_dpi,
            inpaint=args.wm_inpaint,
            use_pdfcpu=(args.wm_pdfcpu.lower() == "true"),
            mask_debug=mask_dir,
        )
        if ok:
            local_pdf = cleaned_pdf
            LOGGER.info(" - PDF limpo será usado na conversão.")
        else:
            LOGGER.warning(" - Limpeza falhou/sem saída. Usando PDF original.")

    LOGGER.info("[2/6] Convertendo com Docling (Markdown + imagens)…")
    md_raw, md_path = convert_with_docling(
        local_pdf,
        work_dir,
        images_scale=2.0,
        include_page_images=bool(args.include_page_images),
    )

    if args.fallback_extractor == "pymupdf4llm" and (not md_raw or len(md_raw) < args.fallback_min_chars):
        LOGGER.info("[2b] Fallback: pymupdf4llm (Docling curto/ruim)…")
        md_fb = convert_fallback_pymupdf4llm(local_pdf)
        if md_fb and len(md_fb) > len(md_raw):
            md_raw = md_fb

    LOGGER.info("[3/6] Higienizando Markdown (whitelist de domínios)…")
    md_clean = sanitize_markdown(md_raw, allowed_domains=allowed_domains)

    LOGGER.info("[3b] Refino do Markdown (headings/linhas/imagens/DTCs)…")
    md_polished = polish_markdown(md_clean)
    md_polished = polish_markdown_fidelity_plus(md_polished)

    brand_terms = [x.strip() for x in (args.drop_brand_terms or "").split(",") if x.strip()]
    md_polished = drop_corporate_branding(md_polished, brand_terms=brand_terms, max_len=args.drop_brand_max_len)

    if args.fidelity_max:
        LOGGER.info("[3b+] Polimento fidelity++ (listas/tabelas/linhas curtas repetidas)…")
        md_polished = polish_markdown_fidelity(
            md_polished,
            drop_repeated_lines=True,
            repeated_min_count=args.drop_repeated_lines_min_count,
            repeated_max_len=args.drop_repeated_lines_max_len,
        )

    LOGGER.info("[3c] Removendo imagens específicas (template/nome/área)…")
    md_filtered, removed_paths = remove_specific_images_from_md(
        md_polished,
        base_dir=md_path.parent,
        template_paths=args.drop_image_template,
        hash_threshold=args.drop_image_hash_threshold,
        name_regex=(args.drop_image_name_re or None),
        max_area=(args.drop_image_max_area if args.drop_image_max_area > 0 else None),
        tpl_phash_threshold=args.drop_image_tpl_phash_threshold,
        aspect_tol=args.drop_image_aspect_tol,
        color_check=bool(args.drop_image_color_check),
        debug=args.drop_image_debug,
    )
    if removed_paths:
        LOGGER.info(" - %s imagem(ns) removida(s).", len(removed_paths))
    md_polished = md_filtered

    if args.drop_repeated:
        LOGGER.info("[3d] Removendo imagens repetidas por conteúdo…")
        area_min = args.drop_repeated_area_min if args.drop_repeated_area_min > 0 else None
        area_max = args.drop_repeated_area_max if args.drop_repeated_area_max > 0 else None
        md_dedup, rep_report = drop_repeated_images_in_md(
            md_polished,
            base_dir=md_path.parent,
            hash_threshold=args.drop_repeated_hash_threshold,
            min_count=args.drop_repeated_min_count,
            size_tol=args.drop_repeated_size_tol,
            keep=args.drop_repeated_keep,
            area_min=area_min,
            area_max=area_max,
            debug=args.drop_repeated_debug,
        )
        if rep_report:
            total = sum(v["count"] for v in rep_report.values())
            LOGGER.info(" - clusters repetidos: %s (ocorrências totais=%s)", len(rep_report), total)
        md_polished = md_dedup

    LOGGER.info("[4/6] Coletando imagens locais referenciadas no MD…")
    local_images = collect_local_images_from_md(md_polished, base_dir=md_path.parent)

    LOGGER.info("[5/6] Enviando imagens ao Supabase… (bucket=%s, pasta=%s/%s/)",
                args.bucket, args.storage_prefix, pdf_id)
    client = supabase_client_from_env()
    url_map = upload_images_to_supabase(client, args.bucket, pdf_id, local_images, storage_prefix=args.storage_prefix)

    LOGGER.info("[6/6] Reescrevendo referências de imagem no Markdown…")
    md_final = rewrite_image_paths_in_md(md_polished, base_dir=md_path.parent, url_map=url_map)

    if args.emit_file:
        md_out = work_dir / f"{pdf_id}.md"
        md_out.write_text(md_final, encoding="utf-8")
        LOGGER.info("Markdown salvo em: %s", md_out)

    return md_final, work_dir

def run_db_batch(args) -> None:
    client = supabase_client_from_env()
    table = args.db_table.strip()
    id_col = args.db_id_col.strip()
    url_col = args.db_url_col.strip()
    if not table or not id_col or not url_col:
        raise RuntimeError("Parâmetros DB inválidos: --db-table, --db-id-col, --db-url-col são obrigatórios.")

    LOGGER.info("[db] Lendo linhas de %s…", table)
    q = client.table(table).select("*")

    if args.db_where_col and args.db_where_val:
        q = q.eq(args.db_where_col, args.db_where_val)

    if args.db_order_col:
        q = q.order(args.db_order_col, desc=(args.db_order_dir == "desc"))

    if args.db_limit and args.db_limit > 0:
        start = max(0, int(args.db_offset or 0))
        end = start + args.db_limit - 1
        q = q.range(start, end)

    resp = q.execute()
    rows: list[dict[str, Any]] = getattr(resp, "data", None) or resp.get("data", [])
    if not rows:
        LOGGER.warning("[db] Nenhuma linha encontrada com os filtros informados.")
        return

    LOGGER.info("[db] %s linha(s) para processar.", len(rows))

    for idx, row in enumerate(rows, 1):
        try:
            row_id = row[id_col]
            pdf_url = row[url_col]
        except KeyError:
            LOGGER.warning("[db] Linha sem colunas %s/%s. Pulando.", id_col, url_col)
            continue

        if not pdf_url:
            LOGGER.warning("[db] %s: URL vazia. Pulando.", row_id)
            continue

        LOGGER.info("[db] (%s/%s) id=%s url=%s", idx, len(rows), row_id, pdf_url)
        try:
            md_final, _work_dir = pipeline_process_pdf(str(row_id), str(pdf_url), args)

            md_url = ""
            if args.db_md_to in ("storage", "both"):
                remote_path = f"{args.db_md_storage_prefix.strip().strip('/')}/{slug_filename(str(row_id))}.md"
                md_url = upload_text_to_supabase(
                    client, args.bucket, remote_path, md_final,
                    content_type="text/markdown; charset=utf-8",
                    upsert=True,
                    add_bom=args.force_bom,     # <— aqui
                )


            if args.db_md_to in ("column", "both") and args.db_md_col:
                try:
                    client.table(table).update({args.db_md_col: md_final}).eq(id_col, row_id).execute()
                except Exception as e:
                    LOGGER.warning("[db] Falha ao gravar MD na coluna %s: %s", args.db_md_col, e)

            updates = {}
            if args.db_success_col:
                updates[args.db_success_col] = datetime.now(timezone.utc).isoformat()
            if args.db_status_col:
                updates[args.db_status_col] = "ok"
            if args.db_error_col:
                updates[args.db_error_col] = ""
            if args.db_md_url_col and md_url:
                updates[args.db_md_url_col] = md_url

            if updates:
                client.table(table).update(updates).eq(id_col, row_id).execute()

            if args.stdout:
                print(md_final)

        except Exception as e:
            LOGGER.error("[db] ERRO no id=%s: %s", row_id, e)
            updates = {}
            if args.db_status_col:
                updates[args.db_status_col] = "fail"
            if args.db_error_col:
                msg = str(e)
                updates[args.db_error_col] = msg[:2000]
            if updates:
                try:
                    client.table(table).update(updates).eq(id_col, row_id).execute()
                except Exception as e2:
                    LOGGER.warning("[db] Falha ao atualizar status de erro: %s", e2)

# =========================================================
# CLI
# =========================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ingestão de PDF com Docling -> MD + Supabase")
    # Básico
    p.add_argument("--pdf-url", required=False, help="URL para download do PDF")
    p.add_argument("--pdf-id", required=False, help="ID do PDF (pasta e nome base)")
    p.add_argument("--out-dir", default="out", help="Diretório de trabalho temporário")
    p.add_argument("--bucket", default=os.getenv("SUPABASE_BUCKET", "articles"), help="Bucket do Supabase")
    p.add_argument("--emit-file", action="store_true", help="Também salva um arquivo .md em disco")
    p.add_argument("--storage-prefix", default="articles/images", help="Prefixo de pasta dentro do bucket")
    p.add_argument("--allowed-domains", default="supabase.co", help="Lista de domínios permitidos separados por vírgula")
    p.add_argument("--stdout", action="store_true", help="Imprime o Markdown final no stdout")
    p.add_argument("--log-level", default="WARNING", help="Nível de log (DEBUG, INFO, WARNING, ERROR)")
    p.add_argument("--force-bom", action="store_true",
                help="Prefixa BOM UTF-8 no arquivo .md para forçar decodificação correta em viewers teimosos.")


    # Filtros de imagem específica
    p.add_argument("--drop-image-template", action="append", default=[], help="Caminho(s) de imagem(s) modelo")
    p.add_argument("--drop-image-hash-threshold", type=int, default=6, help="Limite Hamming aHash (0-64)")
    p.add_argument("--drop-image-name-re", default="", help="Regex de nome de arquivo a remover")
    p.add_argument("--drop-image-max-area", type=int, default=0, help="Remove imagens com área <= px² (0 desativa)")
    p.add_argument("--drop-image-debug", action="store_true", help="Logs de debug do filtro de imagens específicas")
    p.add_argument("--drop-image-tpl-phash-threshold", type=int, default=22, help="Limiar pHash (0-64) p/ templates")
    p.add_argument("--drop-image-aspect-tol", type=float, default=0.20, help="Tolerância relativa de razão de aspecto")
    p.add_argument("--drop-image-color-check", action="store_true", help="Exigir cores da bandeira para dropar por template")

    # Deduplicação de imagens
    p.add_argument("--drop-repeated", action="store_true", help="Remove imagens repetidas por conteúdo")
    p.add_argument("--drop-repeated-min-count", type=int, default=3, help="Mínimo de ocorrências p/ considerar repetida")
    p.add_argument("--drop-repeated-hash-threshold", type=int, default=10, help="Limiar Hamming aHash (0-64)")
    p.add_argument("--drop-repeated-size-tol", type=float, default=0.10, help="Tolerância de tamanho relativo")
    p.add_argument("--drop-repeated-keep", choices=["none", "first"], default="none", help="Se mantém a 1ª ocorrência")
    p.add_argument("--drop-repeated-area-min", type=int, default=20000, help="Considera imagens com área >= px² (0 desativa)")
    p.add_argument("--drop-repeated-area-max", type=int, default=0, help="Considera imagens com área <= px² (0 desativa)")
    p.add_argument("--drop-repeated-debug", action="store_true", help="Logs detalhados dos clusters/remoções")

    # Watermark cleaner
    p.add_argument("--wm-clean", action="store_true", help="Executa limpeza de marcas d’água (remove.py)")
    p.add_argument("--wm-runner", choices=["native", "subprocess"], default="native", help="Modo de execução do cleaner")
    p.add_argument("--wm-script", default="remove.py", help="Caminho do remove.py (quando subprocess)")
    p.add_argument("--wm-python", default=sys.executable, help="Python para subprocess")
    p.add_argument("--wm-dpi", type=int, default=220, help="DPI para raster do fallback de inpainting")
    p.add_argument("--wm-inpaint", choices=["telea", "ns"], default="telea", help="Método de inpainting (OpenCV)")
    p.add_argument("--wm-pdfcpu", choices=["true", "false"], default="true", help="Usar pdfcpu se disponível")
    p.add_argument("--wm-mask-debug", default="", help="Diretório para salvar máscaras (debug)")

    # Polimento/branding
    p.add_argument("--drop-brand-terms", default="", help="Termos de marca/empresa (ex.: 'certta,brasilautomatico').")
    p.add_argument("--drop-brand-max-len", type=int, default=140, help="Comprimento máximo da linha para candidato a branding.")
    p.add_argument("--include-page-images", action="store_true", help="Mantém imagens de página inteira no Docling.")
    p.add_argument("--fallback-extractor", choices=["", "pymupdf4llm"], default="", help="Fallback quando Docling for curto/ruim.")
    p.add_argument("--fallback-min-chars", type=int, default=1200, help="Mínimo de chars para aceitar Docling.")
    p.add_argument("--fidelity-max", action="store_true", help="Liga polimento extra (listas/tabelas/linhas curtas repetidas).")
    p.add_argument("--drop-repeated-lines-min-count", type=int, default=6, help="Min de ocorrências p/ linha curta repetida (fidelity-max).")
    p.add_argument("--drop-repeated-lines-max-len", type=int, default=120, help="Tamanho máximo de linha elegível (fidelity-max).")

    # Modo DB (batch)
    p.add_argument("--db-table", default="", help="Nome da tabela com os PDFs (ativa modo DB se definido)")
    p.add_argument("--db-id-col", default="id", help="Coluna usada como pdf_id")
    p.add_argument("--db-url-col", default="pdf_url", help="Coluna com a URL do PDF")
    p.add_argument("--db-where-col", default="", help="(Opcional) Coluna para filtro simples .eq()")
    p.add_argument("--db-where-val", default="", help="(Opcional) Valor para filtro simples .eq()")
    p.add_argument("--db-limit", type=int, default=0, help="Limite de linhas (0 = sem limite)")
    p.add_argument("--db-offset", type=int, default=0, help="Offset inicial (padrão 0)")
    p.add_argument("--db-order-col", default="", help="(Opcional) Coluna para ordenar")
    p.add_argument("--db-order-dir", choices=["asc", "desc"], default="asc", help="Direção da ordenação")
    p.add_argument("--db-md-to", choices=["storage", "column", "both", "none"], default="storage", help="Onde gravar o MD")
    p.add_argument("--db-md-storage-prefix", default="articles/md", help="Prefixo no bucket para salvar .md")
    p.add_argument("--db-md-col", default="", help="Coluna TEXT para gravar o Markdown (column/both)")
    p.add_argument("--db-md-url-col", default="", help="Coluna para gravar a URL pública do .md (storage/both)")
    p.add_argument("--db-success-col", default="", help="Coluna para marcar sucesso (timestamp ISO)")
    p.add_argument("--db-status-col", default="", help="Coluna para status textual (ok/fail)")
    p.add_argument("--db-error-col", default="", help="Coluna para mensagem de erro (vazio no sucesso)")
    return p

def main():
    load_dotenv()
    parser = build_arg_parser()
    args = parser.parse_args()
    setup_logging(args.log_level)

    if args.db_table:
        run_db_batch(args)
        return

    if not args.pdf_url or not args.pdf_id:
        raise RuntimeError("Uso single-run: informe --pdf-url e --pdf-id, ou use --db-table para modo batch.")

    md_final, _work_dir = pipeline_process_pdf(args.pdf_id, args.pdf_url, args)

    if args.stdout:
        print(md_final)

# =========================================================
# Entrypoint
# =========================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        LOGGER.error("%s", e)
        sys.exit(1)
