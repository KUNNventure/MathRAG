"""Hierarchical Splitter for math textbooks.

按章/节层级切分教材 Markdown 文本，每节 = 一个 chunk，并携带章节 metadata。
当前未启用（RecursiveSplitter 替代）。保留作为实验 1 对照组。

章引言（章标题→第一节标题之间）= 独立 chunk。
阅读与思考 / 数学活动 / 小结 = 各自独立 chunk。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from src.libs.splitter.base_splitter import BaseSplitter

_CN_DIGITS: Dict[str, int] = {
    "零": 0, "〇": 0,
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9,
}


def _cn_to_int(cn: str) -> int:
    cn = cn.strip()
    if not cn:
        return 0
    if cn.isdigit():
        return int(cn)
    if cn in _CN_DIGITS:
        return _CN_DIGITS[cn]
    if cn == "十":
        return 10
    if cn.startswith("十") and len(cn) == 2:
        return 10 + _CN_DIGITS.get(cn[1], 0)
    if cn.endswith("十") and len(cn) == 2:
        return _CN_DIGITS.get(cn[0], 0) * 10
    if "十" in cn:
        tens_s, ones_s = cn.split("十", 1)
        return _CN_DIGITS.get(tens_s, 0) * 10 + _CN_DIGITS.get(ones_s, 0)
    return 0


CHAPTER_RE = re.compile(
    r"^#{0,6}\s*第([一二三四五六七八九十百]+)章\s+(.+)$",
    re.MULTILINE,
)
SECTION_RE = re.compile(r"(\d+(?:\.\d+){1,2})\s*(.+?)$", re.MULTILINE)
_TITLE_CLEAN_RE = re.compile(
    r"((?:\d+(?:\.\d+){1,2}\s*\S)|(?:第[一二三四五六七八九十百]+章))"
)
_PAGE_NUM_RE = re.compile(r"^\s*\d{1,4}\s*$")

SPECIAL_SECTIONS = {
    "阅读与思考": "reading",
    "数学活动": "activity",
    "小结": "summary",
}


class HierarchicalSplitter(BaseSplitter):
    """按教材章/节层级切分文本。当前未启用，保留作为实验对照组。"""

    def __init__(self, settings: Any, **kwargs: Any) -> None:
        self._settings = settings
        self._last_metadata: List[Dict[str, Any]] = []

    def split_text(
        self, text: str, trace: Optional[Any] = None, **kwargs: Any,
    ) -> List[str]:
        self.validate_text(text)
        self._last_metadata = []

        grade = kwargs.get("grade")
        volume = kwargs.get("volume")

        body_start = self._find_body_start(text)
        chapters = self._split_chapters(text, body_start)

        all_chunks: List[Tuple[str, Dict[str, Any]]] = []
        for ch_info in chapters:
            section_chunks = self._split_sections(ch_info, grade=grade, volume=volume)
            all_chunks.extend(section_chunks)

        texts: List[str] = []
        metas: List[Dict[str, Any]] = []
        for text_part, meta in all_chunks:
            texts.append(text_part)
            metas.append(meta)

        self._last_metadata = metas
        self.validate_chunks(texts)
        return texts

    def get_chunk_metadata(self) -> List[Dict[str, Any]]:
        return self._last_metadata

    # ── TOC detection ──
    @staticmethod
    def _is_toc_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        ch_count = len(CHAPTER_RE.findall(stripped))
        sec_count = len(SECTION_RE.findall(stripped))
        if ch_count + sec_count >= 5:
            return True
        if (CHAPTER_RE.match(stripped) or SECTION_RE.match(stripped)):
            if re.search(r"[.…\s]+\d{1,4}\s*$", stripped):
                return True
        return False

    def _find_body_start(self, text: str) -> int:
        lines = text.split("\n")
        if len(lines) < 50:
            return 0
        scan_n = min(40, len(lines))
        toc_indices: List[int] = []
        for i in range(scan_n):
            if self._is_toc_line(lines[i]):
                toc_indices.append(i)
        if len(toc_indices) < 1:
            return 0
        if toc_indices[0] > 10:
            return 0
        last_toc = toc_indices[-1]
        for i in range(scan_n, min(scan_n + 15, len(lines))):
            if self._is_toc_line(lines[i]):
                last_toc = i
        body_line = last_toc + 2
        if body_line < len(lines):
            offset = sum(len(lines[j]) + 1 for j in range(body_line))
            return min(offset, len(text))
        return 0

    # ── Chapter splitting ──
    @staticmethod
    def _clean_chapter_title(raw_title: str) -> str:
        m = _TITLE_CLEAN_RE.search(raw_title)
        if m:
            return raw_title[:m.start()].strip()
        return raw_title.strip()

    def _split_chapters(
        self, text: str, body_start: int = 0
    ) -> List[Dict[str, Any]]:
        body = text[body_start:]
        matches = list(CHAPTER_RE.finditer(body))
        if not matches:
            return [{"chapter_num": 0, "chapter_title": "", "body": body, "offset": body_start}]

        chapter_boundaries: List[Tuple[int, int, str]] = []
        for m in matches:
            chapter_num = _cn_to_int(m.group(1))
            chapter_title = self._clean_chapter_title(m.group(2).strip())
            if not chapter_title:
                continue
            if chapter_boundaries and chapter_boundaries[-1][1] == chapter_num:
                continue
            chapter_boundaries.append((m.start(), chapter_num, chapter_title))

        chapters: List[Dict[str, Any]] = []
        for i, (start, chapter_num, chapter_title) in enumerate(chapter_boundaries):
            end = chapter_boundaries[i + 1][0] if i + 1 < len(chapter_boundaries) else len(body)
            chapters.append({
                "chapter_num": chapter_num, "chapter_title": chapter_title,
                "body": body[start:end].strip(), "offset": body_start + start,
            })
        return chapters

    # ── Section splitting ──
    @staticmethod
    def _find_section_boundaries_in_text(body: str) -> List[Tuple[int, str, str]]:
        boundaries: List[Tuple[int, str, str]] = []
        for m in SECTION_RE.finditer(body):
            sec_num = m.group(1)
            sec_title = m.group(2).strip()
            if len(sec_title) < 2:
                continue
            pos = m.start()
            prefix = body[max(0, pos - 4):pos]
            if re.search(r'[图如见]|见表|参见', prefix):
                continue
            boundaries.append((pos, sec_num, sec_title))
        return boundaries

    def _split_sections(
        self, chapter_info: Dict[str, Any],
        grade: Optional[str] = None, volume: Optional[str] = None,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        chapter_num: int = chapter_info["chapter_num"]
        chapter_title: str = chapter_info["chapter_title"]
        body: str = chapter_info["body"]

        raw_boundaries = self._find_section_boundaries_in_text(body)
        special_boundaries: List[Tuple[int, str, str]] = []
        for keyword, ctype in SPECIAL_SECTIONS.items():
            for m in re.finditer(re.escape(keyword), body):
                ctx_start = max(0, m.start() - 2)
                ctx_end = min(len(body), m.end() + 15)
                ctx = body[ctx_start:ctx_end]
                if len(ctx.strip()) <= 25 and keyword in ctx:
                    special_boundaries.append((m.start(), "", ctype))

        all_boundaries: List[Tuple[int, str, str, str]] = []
        for pos, sec_num, sec_title in raw_boundaries:
            all_boundaries.append((pos, sec_num, sec_title, "section"))
        for pos, sec_num, ctype in special_boundaries:
            all_boundaries.append((pos, sec_num, ctype, ctype))

        all_boundaries.sort(key=lambda x: x[0])
        deduped: List[Tuple[int, str, str, str]] = []
        for b in all_boundaries:
            if deduped and abs(b[0] - deduped[-1][0]) < 5:
                continue
            deduped.append(b)

        if not deduped:
            chunk_text = self._clean_body_text(body, chapter_num=chapter_num, keep_first_chapter_title=True)
            meta = self._build_meta(chapter_num, chapter_title, "", "", "chapter_intro", grade, volume)
            return [(chunk_text, meta)]

        chunks: List[Tuple[str, Dict[str, Any]]] = []
        intro_text = body[:deduped[0][0]].strip()
        intro_text = self._clean_body_text(intro_text, chapter_num=chapter_num, keep_first_chapter_title=True)
        if intro_text:
            chunks.append((intro_text, self._build_meta(
                chapter_num, chapter_title, "", "", "chapter_intro", grade, volume)))

        for idx, (pos, sec_num, sec_title, ctype) in enumerate(deduped):
            next_pos = deduped[idx + 1][0] if idx + 1 < len(deduped) else len(body)
            sec_text = body[pos:next_pos].strip()
            sec_text = self._clean_body_text(sec_text, chapter_num=chapter_num)
            if sec_text:
                chunks.append((sec_text, self._build_meta(
                    chapter_num, chapter_title, sec_num, sec_title, ctype, grade, volume)))
        return chunks

    def _clean_body_text(
        self, text: str, chapter_num: int, keep_first_chapter_title: bool = False,
    ) -> str:
        lines = text.split("\n")
        cleaned: List[str] = []
        chapter_title_kept = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if _PAGE_NUM_RE.match(stripped):
                continue
            ch_match = CHAPTER_RE.match(stripped)
            if ch_match:
                current_chapter_num = _cn_to_int(ch_match.group(1))
                if current_chapter_num == chapter_num:
                    if keep_first_chapter_title and not chapter_title_kept:
                        chapter_title_kept = True
                        cleaned.append(stripped)
                    else:
                        remaining = stripped[ch_match.end():].strip()
                        if remaining:
                            cleaned.append(remaining)
                    continue
                cleaned.append(line)
                continue
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    def _build_meta(
        self, chapter_num: int, chapter_title: str,
        section_num: str, section_title: str, content_type: str,
        grade: Optional[str], volume: Optional[str],
    ) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "chapter_num": chapter_num, "chapter_title": chapter_title,
            "content_type": content_type,
        }
        if section_num:
            meta["section_num"] = section_num
        if section_title:
            meta["section_title"] = section_title
        if grade:
            meta["grade"] = grade
        if volume:
            meta["volume"] = volume
        return meta
