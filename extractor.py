"""
Instagram Content Extractor - Core Module

Extracts content from Instagram posts (single & carousel):
- Caption, hashtags, mentions
- Media URLs with optional local download
- Post metadata (date, likes, comments, owner)
- OCR text extraction from image slides
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from io import BytesIO
from urllib.parse import urlparse

import instaloader
import pytesseract
import requests
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pytesseract import Output

DEFAULT_OCR_LANG = "eng"
DEFAULT_OCR_PSM = 6
DEFAULT_OCR_MIN_CONFIDENCE = 30.0
DEFAULT_FETCH_ATTEMPTS = 3
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/133.0.0.0 Safari/537.36"
    )
}


def extract_shortcode(url: str) -> str:
    """Extract the shortcode from an Instagram post URL."""
    path = urlparse(url).path.strip("/")
    match = re.match(r"(?:p|reel|tv)/([A-Za-z0-9_-]+)", path)
    if match:
        return match.group(1)
    raise ValueError(
        f"Could not extract shortcode from URL: {url}\n"
        "Expected: https://www.instagram.com/p/SHORTCODE/"
    )


def extract_post(
    url: str,
    download_media: bool = True,
    output_dir: str = "downloads",
    ocr: bool = False,
    save_json: bool = False,
    ocr_lang: str = DEFAULT_OCR_LANG,
    ocr_psm: int = DEFAULT_OCR_PSM,
    ocr_min_confidence: float = DEFAULT_OCR_MIN_CONFIDENCE,
) -> dict:
    """Extract all content from a public Instagram post."""
    shortcode = extract_shortcode(url)
    post_output_dir = _build_post_output_dir(output_dir, shortcode)
    loader = _create_loader()
    post = _fetch_post(loader, shortcode)

    media_items = _collect_media(post)
    download_map: dict[int, str] = {}
    if download_media:
        download_map = _download_media(loader, media_items, shortcode, post_output_dir)

    slides = _build_slides(media_items, download_map)
    ocr_results: list[dict] = []
    if ocr:
        _ensure_tesseract_available()
        ocr_results = _ocr_images(
            slides=slides,
            ocr_lang=ocr_lang,
            ocr_psm=ocr_psm,
            ocr_min_confidence=ocr_min_confidence,
        )
        _attach_ocr_results(slides, ocr_results)

    post_data = {
        "shortcode": shortcode,
        "url": f"https://www.instagram.com/p/{shortcode}/",
        "post_type": _get_post_type(post),
        "owner": {
            "username": post.owner_username,
            "user_id": post.owner_id,
        },
        "caption": post.caption or "",
        "accessibility_caption": getattr(post, "accessibility_caption", None),
        "hashtags": list(post.caption_hashtags) if post.caption_hashtags else [],
        "mentions": list(post.caption_mentions) if post.caption_mentions else [],
        "date": post.date_utc.isoformat(),
        "date_local": post.date_local.isoformat() if post.date_local else None,
        "likes": post.likes,
        "comments_count": post.comments,
        "media_count": len(media_items),
        "media": media_items,
        "slides": slides,
        "downloaded_files": [download_map[idx] for idx in sorted(download_map)],
        "ocr_text": ocr_results,
        "ocr_combined_text": _combine_ocr_text(ocr_results),
    }

    os.makedirs(post_output_dir, exist_ok=True)

    if ocr:
        txt_path = os.path.join(post_output_dir, f"{shortcode}.ocr.txt")
        with open(txt_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(post_data["ocr_combined_text"])
        post_data["ocr_text_file"] = txt_path

    if save_json:
        json_path = os.path.join(post_output_dir, f"{shortcode}.json")
        with open(json_path, "w", encoding="utf-8") as file_obj:
            json.dump(post_data, file_obj, indent=2, ensure_ascii=False)
        post_data["json_file"] = json_path

    return post_data


def _create_loader() -> instaloader.Instaloader:
    """Create an Instaloader instance with download disabled."""
    return instaloader.Instaloader(
        download_pictures=False,
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        compress_json=False,
        post_metadata_txt_pattern="",
        max_connection_attempts=3,
    )


def _fetch_post(
    loader: instaloader.Instaloader,
    shortcode: str,
    max_attempts: int = DEFAULT_FETCH_ATTEMPTS,
) -> instaloader.Post:
    """Fetch a post with small backoff to smooth over transient Instagram failures."""
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return instaloader.Post.from_shortcode(loader.context, shortcode)
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            time.sleep(attempt)

    assert last_error is not None
    raise last_error


def _get_post_type(post) -> str:
    """Return a human-readable post type string."""
    if post.typename == "GraphSidecar":
        return "carousel"
    return "video" if post.is_video else "image"


def _build_post_output_dir(base_output_dir: str, shortcode: str) -> str:
    """Return the dedicated output directory for one Instagram post."""
    return os.path.join(base_output_dir, shortcode)


def _collect_media(post) -> list[dict]:
    """Collect all media items from a post."""
    items = []

    if post.typename == "GraphSidecar":
        for idx, node in enumerate(post.get_sidecar_nodes(), start=1):
            entry = {
                "index": idx,
                "type": "video" if node.is_video else "image",
                "url": node.video_url if node.is_video else node.display_url,
            }
            if node.is_video:
                entry["thumbnail_url"] = node.display_url
            items.append(entry)
        return items

    entry = {
        "index": 1,
        "type": "video" if post.is_video else "image",
        "url": post.video_url if post.is_video else post.url,
    }
    if post.is_video:
        entry["thumbnail_url"] = post.url
    items.append(entry)
    return items


def _build_slides(media_items: list[dict], download_map: dict[int, str]) -> list[dict]:
    """Build per-slide records to make downstream consumption easier."""
    slides = []
    for item in media_items:
        slide = dict(item)
        slide["file_path"] = download_map.get(item["index"])
        slides.append(slide)
    return slides


def _download_media(
    loader: instaloader.Instaloader,
    media_items: list[dict],
    shortcode: str,
    output_dir: str,
) -> dict[int, str]:
    """Download all media files and return a map of slide index -> file path."""
    os.makedirs(output_dir, exist_ok=True)
    downloaded: dict[int, str] = {}

    for item in media_items:
        ext = "mp4" if item["type"] == "video" else "jpg"
        filename = f"{shortcode}_{item['index']}.{ext}"
        filepath = os.path.join(output_dir, filename)

        if _is_valid_cached_media(item, filepath):
            downloaded[item["index"]] = filepath
            continue

        _remove_invalid_cached_file(filepath)

        try:
            loader.context.get_and_write_raw(item["url"], filepath)
            if _is_valid_cached_media(item, filepath):
                downloaded[item["index"]] = filepath
            else:
                _remove_invalid_cached_file(filepath)
                print(f"  Warning: Downloaded media #{item['index']} was invalid and was discarded")
        except Exception as exc:
            print(f"  Warning: Failed to download media #{item['index']}: {exc}")

    return downloaded


def _is_valid_cached_media(item: dict, filepath: str) -> bool:
    """Check whether an existing cached media file is safe to reuse."""
    if not os.path.exists(filepath):
        return False
    if os.path.getsize(filepath) == 0:
        return False

    if item["type"] == "image":
        try:
            with Image.open(filepath) as image:
                image.verify()
            return True
        except Exception:
            return False

    return os.path.getsize(filepath) >= 1024


def _remove_invalid_cached_file(filepath: str) -> None:
    """Delete a known-bad cache file if it exists."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except OSError:
        pass


def _ensure_tesseract_available() -> None:
    """Raise a clear error if the local Tesseract binary is missing."""
    try:
        pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract is not installed or not available in PATH. "
            "Install it first, for example with `brew install tesseract`."
        ) from exc


def _ocr_images(
    slides: list[dict],
    ocr_lang: str,
    ocr_psm: int,
    ocr_min_confidence: float,
) -> list[dict]:
    """Run OCR on each image slide and return structured OCR results."""
    results = []

    for slide in slides:
        index = slide["index"]
        try:
            image, media_source = _load_ocr_image(slide)
            result = _run_best_ocr(
                image=image,
                lang=ocr_lang,
                psm=ocr_psm,
                min_confidence=ocr_min_confidence,
            )
            results.append(
                {
                    "slide": index,
                    "text": result["text"],
                    "lines": result["lines"],
                    "confidence": result["confidence"],
                    "word_count": result["word_count"],
                    "line_count": result["line_count"],
                    "variant": result["variant"],
                    "media_type": slide["type"],
                    "ocr_source": media_source,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "slide": index,
                    "text": f"[OCR failed: {exc}]",
                    "lines": [],
                    "confidence": 0.0,
                    "word_count": 0,
                    "line_count": 0,
                    "variant": "failed",
                    "media_type": slide["type"],
                    "ocr_source": "failed",
                }
            )

    return results


def _attach_ocr_results(slides: list[dict], ocr_results: list[dict]) -> None:
    """Attach OCR output directly onto slide objects."""
    by_slide = {item["slide"]: item for item in ocr_results}
    for slide in slides:
        if slide["type"] in {"image", "video"}:
            slide["ocr"] = by_slide.get(slide["index"])


def _combine_ocr_text(ocr_results: list[dict]) -> str:
    """Build one plain-text OCR artifact for easy downstream usage."""
    sections = []
    for item in ocr_results:
        text = item["text"].strip()
        if not text or text.startswith("[OCR failed"):
            continue
        sections.append(f"Slide {item['slide']}\n{text}")
    return "\n\n".join(sections)


def _load_ocr_image(slide: dict) -> tuple[Image.Image, str]:
    """Load the best available OCR image for a slide.

    For videos, OCR runs against the cover/thumbnail image.
    """
    file_path = slide.get("file_path")
    if slide["type"] == "image" and file_path and os.path.exists(file_path):
        with Image.open(file_path) as image:
            return image.copy(), "downloaded_image"

    image_url = _get_ocr_image_url(slide)
    response = requests.get(image_url, headers=REQUEST_HEADERS, timeout=20)
    response.raise_for_status()
    with Image.open(BytesIO(response.content)) as image:
        return image.copy(), "remote_thumbnail" if slide["type"] == "video" else "remote_image"


def _get_ocr_image_url(slide: dict) -> str:
    """Return the OCR-friendly image URL for a slide."""
    if slide["type"] == "video":
        thumbnail_url = slide.get("thumbnail_url")
        if not thumbnail_url:
            raise RuntimeError("Video slide has no thumbnail URL for OCR")
        return thumbnail_url
    return slide["url"]


def _run_best_ocr(
    image: Image.Image,
    lang: str,
    psm: int,
    min_confidence: float,
) -> dict:
    """Try a few OCR-oriented image variants and keep the best result."""
    candidates = []
    for variant_name, variant_image in _build_ocr_variants(image):
        candidate = _extract_text_from_variant(
            image=variant_image,
            lang=lang,
            psm=psm,
            min_confidence=min_confidence,
        )
        candidate["variant"] = variant_name
        candidates.append(candidate)

    return max(candidates, key=_ocr_score)


def _build_ocr_variants(image: Image.Image) -> list[tuple[str, Image.Image]]:
    """Create multiple preprocessed versions to handle different post designs."""
    normalized = image.convert("RGB")
    width, height = normalized.size
    if width < 2200:
        scale = 2200 / width
        normalized = normalized.resize((int(width * scale), int(height * scale)), Image.LANCZOS)

    grayscale = ImageOps.grayscale(normalized)
    grayscale = ImageOps.autocontrast(grayscale)

    enhanced = ImageEnhance.Contrast(grayscale).enhance(1.8)
    sharpened = enhanced.filter(ImageFilter.SHARPEN)
    thresholded = sharpened.point(lambda px: 255 if px > 180 else 0).convert("L")
    softened = sharpened.filter(ImageFilter.MedianFilter(size=3))

    return [
        ("enhanced", sharpened),
        ("thresholded", thresholded),
        ("softened", softened),
    ]


def _extract_text_from_variant(
    image: Image.Image,
    lang: str,
    psm: int,
    min_confidence: float,
) -> dict:
    """Extract OCR lines plus confidence from one preprocessed image."""
    config = f"--oem 3 --psm {psm} -c preserve_interword_spaces=1"
    data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=Output.DICT)

    grouped_lines: dict[tuple[int, int, int], list[str]] = {}
    confidences: list[float] = []

    for idx, raw_text in enumerate(data["text"]):
        text = _normalize_ocr_fragment(raw_text)
        conf = _parse_confidence(data["conf"][idx])
        if not text or conf < min_confidence:
            continue

        key = (
            data["block_num"][idx],
            data["par_num"][idx],
            data["line_num"][idx],
        )
        grouped_lines.setdefault(key, []).append(text)
        confidences.append(conf)

    lines = [
        _normalize_ocr_line(" ".join(words))
        for _, words in sorted(grouped_lines.items())
        if words
    ]
    lines = [line for line in lines if line]

    if not lines:
        fallback = pytesseract.image_to_string(image, lang=lang, config=config)
        lines = [
            _normalize_ocr_line(line)
            for line in fallback.splitlines()
            if _normalize_ocr_line(line)
        ]

    text = "\n".join(lines)
    word_count = sum(len(line.split()) for line in lines)
    confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0

    return {
        "text": text,
        "lines": lines,
        "confidence": confidence,
        "word_count": word_count,
        "line_count": len(lines),
    }


def _ocr_score(result: dict) -> float:
    """Score OCR candidates by balancing confidence and amount of text recovered."""
    if not result["text"]:
        return 0.0
    return result["confidence"] * math.log(result["word_count"] + 2, 2)


def _normalize_ocr_fragment(text: str) -> str:
    """Normalize an OCR fragment while preserving useful punctuation."""
    return re.sub(r"\s+", " ", text).strip()


def _normalize_ocr_line(text: str) -> str:
    """Normalize OCR output lines while preserving readable formatting."""
    return re.sub(r"\s+", " ", text).strip()


def _parse_confidence(raw_value: str) -> float:
    """Parse Tesseract confidence values safely."""
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return -1.0
