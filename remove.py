#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pdf_watermark_cleaner.py

Algoritmo completo para identificar e remover marcas d'água em PDFs:
1) Remoção estrutural (OCG/Artifact/Stamp) com pikepdf
2) (Opcional) pdfcpu watermark/stamp remove via subprocess
3) Raster + inpainting (OpenCV) com detecção automática de máscara (fallback)

Dependências (pip): pikepdf, pymupdf, opencv-python, numpy
Opcional: pdfcpu (binário no PATH)

Uso:
  python pdf_watermark_cleaner.py --input in.pdf --output out.pdf --dpi 220 --inpaint telea --pdfcpu true --mask-debug ./masks
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import fitz  # PyMuPDF
import cv2   # OpenCV
import pikepdf
from pikepdf import Name, Dictionary, Object, Stream


# ----------------------------- Util / Logs -----------------------------

def log(msg: str):
    print(f"[pdfwm] {msg}", flush=True)


def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------- Detecção estrutural -----------------------------

SUSPECT_NAME_RE = re.compile(r"(watermark|marca\s*d[’']?água|draft|rascunho|confiden|background|sample|header|footer|cambio|automatico|brasil)", re.I)

@dataclass
class PageFeature:
    page: int
    has_ocg: bool
    ocg_names: List[str]
    suspect_ocg: bool
    hint_artifact: bool
    annot_watermark: bool
    xobject_suspect: bool


def detect_pdf_features(in_pdf: Path) -> List[PageFeature]:
    feats: List[PageFeature] = []
    with pikepdf.open(str(in_pdf)) as pdf:
        ocprops = pdf.Root.get("/OCProperties", None)
        ocg_names = []
        if ocprops and "/OCGs" in ocprops:
            for ocg in ocprops["/OCGs"]:
                try:
                    name = str(ocg.get("/Name", ""))
                except Exception:
                    name = ""
                if name:
                    ocg_names.append(name)

        for i, page in enumerate(pdf.pages, start=1):
            pf = PageFeature(
                page=i, has_ocg=bool(ocg_names), ocg_names=list(ocg_names),
                suspect_ocg=any(SUSPECT_NAME_RE.search(n or "") for n in ocg_names),
                hint_artifact=False, annot_watermark=False, xobject_suspect=False
            )

            # Annotations (stamps/watermark)
            try:
                annots = page.get("/Annots", None)
                if annots:
                    for a in annots:
                        sub = str(a.get("/Subtype", ""))
                        nm = str(a.get("/NM", "") ) + " " + str(a.get("/T", "") ) + " " + str(a.get("/Name", ""))
                        if "/Stamp" in sub or "/Watermark" in sub or SUSPECT_NAME_RE.search(nm or ""):
                            pf.annot_watermark = True
                            break
            except Exception:
                pass

            # Hint de Artifact/OC no conteúdo
            try:
                ct = page.get_contents()
                buf = ct.read_bytes().decode("latin-1", "ignore") if ct is not None else ""
                if "/Artifact" in buf or "/OC " in buf:
                    pf.hint_artifact = True
                # XObject nomes suspeitos (WM/Watermark)
                if "/XObject" in buf and re.search(r"/(WM|Watermark)[0-9A-Za-z]*\s+Do", buf):
                    pf.xobject_suspect = True
            except Exception:
                pass

            feats.append(pf)
    return feats


# ----------------------------- Remoção estrutural -----------------------------

def _strip_marked_content(text: str) -> str:
    # Remove blocos marcados /Artifact ... EMC e /OC ... EMC
    text = re.sub(r"/Artifact\s+BDC.*?EMC", "", text, flags=re.S)
    text = re.sub(r"/OC\s+[\[/][^\]]*?\]\s+BDC.*?EMC", "", text, flags=re.S)
    # Remove blocos /Artifact específicos de watermark
    text = re.sub(r"/Artifact\s*<</Subtype\s*/Watermark.*?>>BDC.*?EMC", "", text, flags=re.S)
    text = re.sub(r"/Artifact\s*<</Subtype\s*/Header.*?>>BDC.*?EMC", "", text, flags=re.S)
    text = re.sub(r"/Artifact\s*<</Subtype\s*/Footer.*?>>BDC.*?EMC", "", text, flags=re.S)
    # Remove chamadas para XObjects de watermark (Fm0, Fm1, Fm2)
    text = re.sub(r"/Fm[0-9]+\s+Do", "", text)
    # Remove texto específico da marca d'água
    text = re.sub(r"www\.cambioautomaticodobrasil\.com\.br", "", text, flags=re.I)
    text = re.sub(r"CÂMBIO\s+AUTOMÁTICO\s+DO\s+BRASIL", "", text, flags=re.I)
    return text


def _remove_suspect_xobjects(pdf: pikepdf.Pdf, page: pikepdf.Page):
    """Remove XObjects com nomes suspeitos e suas chamadas 'Do' do content stream."""
    try:
        resources = page.get("/Resources", Dictionary())
        xobjs = resources.get("/XObject", Dictionary())
        remove_keys = []
        for key, xobj in xobjs.items():
            try:
                name = str(key)
                # Remove Forms que são suspeitas (Fm0, Fm1, Fm2 são as marcas d'água)
                if re.match(r"/Fm[0-9]+", name):
                    remove_keys.append(key)
                    print(f"[Remoção] XObject Form suspeito encontrado: {name}")
                    continue
                if SUSPECT_NAME_RE.search(name):
                    remove_keys.append(key)
                    print(f"[Remoção] XObject suspeito encontrado: {name}")
                    continue
                # stream subtype /Image com nome sugestivo "WM"
                if re.search(r"/WM[a-zA-Z0-9]*", name):
                    remove_keys.append(key)
                    print(f"[Remoção] XObject WM encontrado: {name}")
            except Exception:
                pass
        if remove_keys:
            print(f"[Remoção] Removendo {len(remove_keys)} XObjects suspeitos")
            for k in remove_keys:
                del xobjs[k]
            resources["/XObject"] = xobjs
            page["/Resources"] = resources

        # Limpa chamadas Do para os XObjects removidos
        try:
            if hasattr(page, 'Contents'):
                cs = page.Contents
                if cs:
                    if hasattr(cs, 'read_bytes'):
                        buf = cs.read_bytes().decode("latin-1", "ignore")
                    else:
                        buf = str(cs)
                    for k in remove_keys:
                        kname = str(k).strip("/")
                        buf = re.sub(rf"/{re.escape(kname)}\s+Do", "", buf)
                    page.Contents = Stream(pdf, buf.encode("latin-1", "ignore"))
        except Exception as e:
            print(f"[Erro] Ao limpar chamadas Do: {e}")
            pass
    except Exception as e:
        print(f"[Erro] Ao processar XObjects: {e}")
        pass


def structural_cleanup(in_pdf: Path, out_pdf: Path) -> Dict:
    """
    Remove: OCG/Artifacts, annotations de stamp/watermark e XObjects suspeitos.
    Retorna um dict com estatísticas simples.
    """
    stats = {
        "pages": 0,
        "removed_annots": 0,
        "rewrote_contents": 0,
        "removed_xobjects": 0,
    }
    with pikepdf.open(str(in_pdf)) as pdf:
        stats["pages"] = len(pdf.pages)

        # Remover OCGs suspeitos do catálogo
        ocprops = pdf.Root.get("/OCProperties", None)
        if ocprops:
            new_ocprops = Dictionary()
            if "/OCGs" in ocprops:
                filtered_ocgs = []
                for ocg in ocprops["/OCGs"]:
                    name = str(ocg.get("/Name", ""))
                    if not SUSPECT_NAME_RE.search(name):
                        filtered_ocgs.append(ocg)
                    else:
                        print(f"[Remoção] OCG removido: {name}")
                new_ocprops["/OCGs"] = pikepdf.Array(filtered_ocgs)
            pdf.Root["/OCProperties"] = new_ocprops

        # Remover referências a OCGs suspeitos em páginas
        for page in pdf.pages:
            try:
                annots = page.get("/Annots", None)
                if annots:
                    new_annots = []
                    removed_any = False
                    for a in annots:
                        sub = str(a.get("/Subtype", ""))
                        nm = (str(a.get("/NM", "") ) + " " + str(a.get("/T", "") ) + " " + str(a.get("/Name", ""))).strip()
                        if "/Stamp" in sub or "/Watermark" in sub or SUSPECT_NAME_RE.search(nm or ""):
                            removed_any = True
                            continue
                        new_annots.append(a)
                    if removed_any:
                        page["/Annots"] = pikepdf.Array(new_annots)
                    else:
                        try:
                            del page["/Annots"]
                        except Exception:
                            pass
            except Exception:
                pass

        # Remover /OCProperties se vazio
        if pdf.Root.get("/OCProperties", None) is None or not pdf.Root["/OCProperties"]:
            try:
                del pdf.Root["/OCProperties"]
            except Exception:
                pass

        # Coleta OCGs suspeitos (não necessário para exclusão direta de BDC/EMC, mas útil)
        # Removido devido a problemas de compatibilidade com pikepdf

        for i, page in enumerate(pdf.pages, start=1):
            # 1) Remover annotations suspeitas
            annots = page.get("/Annots", None)
            if annots:
                new_annots = []
                removed_any = False
                for a in annots:
                    sub = str(a.get("/Subtype", ""))
                    nm = (str(a.get("/NM", "") ) + " " + str(a.get("/T", "") ) + " " + str(a.get("/Name", ""))).strip()
                    if "/Stamp" in sub or "/Watermark" in sub or SUSPECT_NAME_RE.search(nm or ""):
                        removed_any = True
                        print(f"[Remoção] Annotation removida na página {i}: {sub} - {nm}")
                        continue
                    new_annots.append(a)
                if removed_any:
                    stats["removed_annots"] += 1
                    if new_annots:
                        page["/Annots"] = pikepdf.Array(new_annots)
                    else:
                        try:
                            del page["/Annots"]
                        except Exception:
                            pass

            # 2) Reescrever content stream removendo blocos /Artifact e /OC ... EMC
            try:
                if hasattr(page, 'Contents'):
                    cs = page.Contents
                    if cs:
                        if hasattr(cs, 'read_bytes'):
                            buf = cs.read_bytes().decode("latin-1", "ignore")
                        else:
                            buf = str(cs)
                        new_buf = _strip_marked_content(buf)
                        if new_buf != buf:
                            stats["rewrote_contents"] += 1
                            print(f"[Remoção] Content stream reescrito na página {i}")
                            page.Contents = Stream(pdf, new_buf.encode("latin-1", "ignore"))
            except Exception as e:
                print(f"[Erro] Content stream página {i}: {e}")
                pass

            # 3) Remover XObjects suspeitos e respectivas chamadas
            before = page.get("/Resources", Dictionary()).get("/XObject", Dictionary())
            before_count = len(before.keys()) if hasattr(before, "keys") else 0
            _remove_suspect_xobjects(pdf, page)
            after = page.get("/Resources", Dictionary()).get("/XObject", Dictionary())
            after_count = len(after.keys()) if hasattr(after, "keys") else 0
            if after_count < before_count:
                stats["removed_xobjects"] += (before_count - after_count)
                print(f"[Remoção] {before_count - after_count} XObjects removidos na página {i}")

        pdf.save(str(out_pdf))
    return stats


# ----------------------------- pdfcpu (opcional) -----------------------------

def pdfcpu_remove(in_pdf: Path, out_pdf: Path) -> bool:
    """
    Se pdfcpu estiver no PATH, tenta:
      pdfcpu watermark remove in.pdf out.pdf
    Retorna True se executou e gerou out_pdf, False caso contrário.
    """
    if which("pdfcpu") is None:
        return False
    try:
        cmd = ["pdfcpu", "watermark", "remove", str(in_pdf), str(out_pdf)]
        log("Executando pdfcpu watermark remove ...")
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if r.returncode == 0 and out_pdf.exists() and out_pdf.stat().st_size > 0:
            return True
        # Alguns PDFs usam "stamp"
        cmd = ["pdfcpu", "stamp", "remove", str(in_pdf), str(out_pdf)]
        log("Executando pdfcpu stamp remove ...")
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return (r.returncode == 0 and out_pdf.exists() and out_pdf.stat().st_size > 0)
    except Exception as e:
        log(f"pdfcpu falhou: {e}")
        return False


# ----------------------------- Raster + Inpainting -----------------------------

def render_pdf_to_images(in_pdf: Path, out_dir: Path, dpi: int = 220) -> List[Path]:
    ensure_dir(out_dir)
    images: List[Path] = []
    with fitz.open(str(in_pdf)) as doc:
        for i, page in enumerate(doc, start=1):
            zoom = dpi / 72.0
            rot = int(getattr(page, "rotation", 0) or 0)
            mat = fitz.Matrix(zoom, zoom)
            # compat: PyMuPDF novo (prerotate) e antigo (preRotate)
            try:
                mat = mat.prerotate(rot)
            except AttributeError:
                mat = mat.preRotate(rot)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pth = out_dir / f"page_{i:04d}.png"
            pix.save(str(pth))
            images.append(pth)
    return images


def auto_watermark_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Heurística robusta, mas simples, para detectar marcas d'água textuais/fantasma:
    - converte para cinza
    - realça traços finos com blackhat/top-hat
    - threshold adaptativo
    - filtra por componentes muito pequenos/muito grandes
    Retorna máscara binária uint8 {0,255}
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Normaliza contraste (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    gray = clahe.apply(gray)

    # Enfatiza traços (top-hat e black-hat)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    toph = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    blkh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)
    enh = cv2.addWeighted(toph, 0.5, blkh, 0.5, 0)

    # Threshold adaptativo + Otsu de reforço
    thr_adapt = cv2.adaptiveThreshold(enh, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 31, 5)
    _, thr_otsu = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_or(thr_adapt, thr_otsu)

    # Remove ruído pequeno
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Fecha furos e consolida texto largo (muitas marcas d'água são grandes)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2, iterations=1)

    # Filtra por área de componente conectada (mantém componentes de 0.05% a 60% da imagem)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H, W = mask.shape
    area = H * W
    keep = np.zeros_like(mask)
    for lbl in range(1, num_labels):
        a = stats[lbl, cv2.CC_STAT_AREA]
        if 0.0005 * area <= a <= 0.60 * area:
            keep[labels == lbl] = 255

    kedge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    keep = cv2.erode(keep, kedge, iterations=1)          # encolhe 1px para longe das letras
    keep = (keep > 0).astype(np.uint8) * 255
    print(f"[Debug] Máscara detectada: {np.sum(keep > 0)} pixels de {keep.size} total ({np.sum(keep > 0)/keep.size*100:.3f}%)")
    return keep
    
    return keep


def inpaint_image(bgr: np.ndarray, mask: np.ndarray, method: str = "telea") -> np.ndarray:
    """
    Inpaint com OpenCV: 'telea' (rápido, bom para texto) ou 'ns' (Navier-Stokes).
    """
    radius = 3
    if method.lower() == "ns":
        flag = cv2.INPAINT_NS
    else:
        flag = cv2.INPAINT_TELEA
    # Necessário: mask 1 canal 8-bit {0,255}
    mask_bin = (mask > 127).astype(np.uint8) * 255
    result = cv2.inpaint(bgr, mask_bin, inpaintRadius=radius, flags=flag)
    return result


def rebuild_pdf_from_mix(original_pdf: Path, inpainted_pages: Dict[int, Path], out_pdf: Path):
    """
    Cria um novo PDF:
    - Se a página foi inpainted -> insere a imagem rasterizada (perde o texto daquela página)
    - Caso contrário -> copia a página original
    """
    out = fitz.open()
    with fitz.open(str(original_pdf)) as src:
        for i, page in enumerate(src, start=1):
            if i in inpainted_pages:
                # Cria página do mesmo tamanho e insere a imagem
                rect = page.rect
                newp = out.new_page(width=rect.width, height=rect.height)
                img_path = str(inpainted_pages[i])
                newp.insert_image(rect, filename=img_path, keep_proportion=False)
            else:
                out.insert_pdf(src, from_page=i-1, to_page=i-1)
    out.save(str(out_pdf))
    out.close()


# ----------------------------- Pipeline principal -----------------------------

def process_pdf(
    in_pdf: Path,
    out_pdf: Path,
    dpi: int = 220,
    inpaint_method: str = "telea",
    use_pdfcpu: bool = True,
    mask_debug_dir: Optional[Path] = None,
    protect_text: bool = True, 
    protect_pad: int = 3
) -> Dict:
    """
    Executa pipeline em camadas:
     1) Remoção estrutural pikepdf
     2) (opcional) pdfcpu
     3) Raster + inpaint se ainda houver indicação (ou sempre tenta e só aplica se máscara tiver "massa")
    Retorna relatório JSON com estatísticas.
    """
    report = {"input": str(in_pdf), "output": str(out_pdf), "steps": []}
    tmp = Path(tempfile.mkdtemp(prefix="pdfwm_"))
    try:
        # Passo 0: detectar
        feats = detect_pdf_features(in_pdf)
        report["features_before"] = [f.__dict__ for f in feats]

        # Passo 1: estrutural
        p1_out = tmp / "structural.pdf"
        stats1 = structural_cleanup(in_pdf, p1_out)
        report["steps"].append({"structural_cleanup": stats1})

        work_pdf = p1_out if p1_out.exists() and p1_out.stat().st_size > 0 else in_pdf

        # Passo 2: pdfcpu (se habilitado e instalado)
        if use_pdfcpu and which("pdfcpu"):
            p2_out = tmp / "pdfcpu.pdf"
            ok = pdfcpu_remove(work_pdf, p2_out)
            if ok:
                report["steps"].append({"pdfcpu": "applied"})
                work_pdf = p2_out
            else:
                report["steps"].append({"pdfcpu": "skipped_or_failed"})
        else:
            report["steps"].append({"pdfcpu": "not_available_or_disabled"})

        img_dir = tmp / "raster"
        imgs = render_pdf_to_images(work_pdf, img_dir, dpi=dpi)

        inpainted_pages: Dict[int, Path] = {}
        if mask_debug_dir:
            ensure_dir(mask_debug_dir)

        # Abra o PDF para gerar as máscaras de texto alinhadas
        doc = fitz.open(str(work_pdf))

        for idx, img_path in enumerate(imgs, start=1):
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue

            wm_mask = auto_watermark_mask(bgr)

            if protect_text:
                tmask = text_protect_mask_for_page(doc, idx - 1, (bgr.shape[0], bgr.shape[1]), dpi, pad=protect_pad)
                # remove áreas de texto da máscara de watermark
                wm_mask = cv2.bitwise_and(wm_mask, cv2.bitwise_not(tmask))
                if mask_debug_dir:
                    cv2.imwrite(str(Path(mask_debug_dir) / f"textmask_{idx:04d}.png"), tmask)

            coverage = (wm_mask > 0).sum() / wm_mask.size

            # se após proteger o texto quase não sobrar máscara, não inpaint
            if coverage >= 0.0001:
                out_img = inpaint_image(bgr, wm_mask, method=inpaint_method)
                out_path = img_dir / f"page_{idx:04d}_inpaint.png"
                cv2.imwrite(str(out_path), out_img)
                inpainted_pages[idx] = out_path

            if mask_debug_dir:
                cv2.imwrite(str(Path(mask_debug_dir) / f"mask_{idx:04d}.png"), wm_mask)

        doc.close()
        if inpainted_pages:
            final_out = tmp / "final.pdf"
            rebuild_pdf_from_mix(work_pdf, inpainted_pages, final_out)
            shutil.copyfile(final_out, out_pdf)
            report["steps"].append({"raster_inpaint": f"applied_to_{len(inpainted_pages)}_pages"})
        else:
            # Sem páginas exigindo inpaint -> saída é work_pdf
            shutil.copyfile(work_pdf, out_pdf)
            report["steps"].append({"raster_inpaint": "no_pages_needed"})

        # Passo 4: detectar novamente (pós)
        feats_after = detect_pdf_features(out_pdf)
        report["features_after"] = [f.__dict__ for f in feats_after]

        return report
    finally:
        # mantenha tmp para debug? Se quiser manter, comente a linha abaixo.
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------- Função de análise para debug -----------------------------

def analyze_pdf(in_pdf: Path):
    """Analisa e imprime estrutura relevante do PDF para depuração."""
    with pikepdf.open(str(in_pdf)) as pdf:
        print("[Análise] Total de páginas:", len(pdf.pages))
        
        # OCGs globais
        ocprops = pdf.Root.get("/OCProperties", None)
        if ocprops:
            print("[Análise] OCGs encontrados:")
            for ocg in ocprops.get("/OCGs", []):
                name = str(ocg.get("/Name", "Sem nome"))
                print(f"  - OCG: {name}")
        
        # Por página
        for i, page in enumerate(pdf.pages, start=1):
            print(f"\n[Análise] Página {i}:")
            
            # Annotations
            annots = page.get("/Annots", None)
            if annots:
                print("  Annotations:")
                for a in annots:
                    sub = str(a.get("/Subtype", "Desconhecido"))
                    nm = str(a.get("/NM", "")) + " " + str(a.get("/T", "")) + " " + str(a.get("/Name", ""))
                    print(f"    - Subtype: {sub}, Name: {nm}")
            
            # XObjects
            resources = page.get("/Resources", Dictionary())
            xobjs = resources.get("/XObject", Dictionary())
            if xobjs:
                print("  XObjects:")
                for key, xobj in xobjs.items():
                    name = str(key)
                    subtype = str(xobj.get("/Subtype", "Desconhecido"))
                    print(f"    - {name}: Subtype {subtype}")
            
            # Trecho do content stream (primeiros 500 chars para visão geral)
            try:
                if hasattr(page, 'Contents'):
                    cs = page.Contents
                    if cs:
                        if hasattr(cs, 'read_bytes'):
                            buf = cs.read_bytes().decode("latin-1", "ignore")[:500]
                        else:
                            buf = str(cs)[:500]
                        print("  Content Stream (trecho):")
                        print(buf.replace("\n", " ").replace("\r", " "))
                    else:
                        print("  Sem content stream.")
                else:
                    print("  Sem content stream.")
            except Exception as e:
                print(f"  Erro ao ler content stream: {e}")

# ----------------------------- CLI -----------------------------

def text_protect_mask_for_page(doc: fitz.Document, page_index: int, shape_hw: Tuple[int, int],
                               dpi: int, pad: int = 3) -> np.ndarray:
    """
    Gera máscara binária (uint8) 255 onde há TEXTO (com margem 'pad' em pixels).
    Alinha com a rasterização feita com zoom=dpi/72 e respeita rotação da página.
    """
    H, W = shape_hw  # (altura, largura) da imagem BGR
    mask = np.zeros((H, W), dtype=np.uint8)

    page = doc[page_index]
    zoom = dpi / 72.0
    rot = page.rotation or 0  # garante alinhamento com get_pixmap()
    # As coords de get_text("words") já estão no espaço do page.rect com rotação aplicada pelo MuPDF

    words = page.get_text("words") or []
    # words: [x0, y0, x1, y1, "word", block_no, line_no, word_no]
    for w in words:
        x0, y0, x1, y1 = w[0], w[1], w[2], w[3]
        # escala para pixels
        x0p = int(x0 * zoom) - pad
        y0p = int(y0 * zoom) - pad
        x1p = int(x1 * zoom) + pad
        y1p = int(y1 * zoom) + pad
        # clamp
        x0p = max(0, min(W - 1, x0p))
        x1p = max(0, min(W - 1, x1p))
        y0p = max(0, min(H - 1, y0p))
        y1p = max(0, min(H - 1, y1p))
        if x1p > x0p and y1p > y0p:
            cv2.rectangle(mask, (x0p, y0p), (x1p, y1p), 255, thickness=-1)

    # Dilatação mínima para pegar acentos/serifas
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(mask, k, iterations=1)
    return mask


def parse_args():
    ap = argparse.ArgumentParser(description="Removedor de marcas d'água em PDF (3 camadas).")
    ap.add_argument("--input", "-i", required=True, help="Caminho do PDF de entrada")
    ap.add_argument("--output", "-o", default=None, help="Caminho do PDF de saída (obrigatório se não for --analyze)")
    ap.add_argument("--dpi", type=int, default=220, help="DPI para rasterização do fallback (200-300 recomendado)")
    ap.add_argument("--inpaint", choices=["telea", "ns"], default="telea", help="Método de inpainting OpenCV")
    ap.add_argument("--pdfcpu", type=str, default="true", help="Usar pdfcpu se disponível? true/false")
    ap.add_argument("--mask-debug", type=str, default=None, help="Diretório para salvar máscaras detectadas (debug)")
    ap.add_argument("--analyze", action="store_true", help="Analisar estrutura do PDF sem processar")
        # ... em parse_args()
    ap.add_argument("--protect-text", action="store_true", default=True,
                    help="Protege áreas de texto durante o inpaint (recomendado).")
    ap.add_argument("--protect-pad", type=int, default=4,
                    help="Margem (px) ao redor do texto para proteger.")

    return ap.parse_args()

def main():
    args = parse_args()
    in_pdf = Path(args.input).resolve()
    
    if args.analyze:
        if not in_pdf.exists():
            log(f"Arquivo não encontrado: {in_pdf}")
            sys.exit(2)
        analyze_pdf(in_pdf)
        return
    
    # Resto do main original
    out_pdf = Path(args.output).resolve() if args.output else None
    if not out_pdf:
        log("Forneça --output para processamento.")
        sys.exit(1)
    mask_debug = Path(args.mask_debug).resolve() if args.mask_debug else None
    use_pdfcpu = str(args.pdfcpu).lower() in ("1", "true", "yes", "y")
    
    if not in_pdf.exists():
        log(f"Arquivo não encontrado: {in_pdf}")
        sys.exit(2)
    
    ensure_dir(out_pdf.parent)
    
    log("Iniciando pipeline...")
    report = process_pdf(
        in_pdf=in_pdf,
        out_pdf=out_pdf,
        dpi=args.dpi,
        inpaint_method=args.inpaint,
        use_pdfcpu=use_pdfcpu,
        mask_debug_dir=mask_debug,
        protect_text=args.protect_text,
        protect_pad=args.protect_pad
    )

    
    log("Concluído.")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()