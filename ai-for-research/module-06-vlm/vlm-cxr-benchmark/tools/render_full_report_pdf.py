#!/usr/bin/env python3
"""Render the verified Thai Markdown experiment report as a polished A4 PDF."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    CondPageBreak,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


FONT_REGULAR = "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/noto/NotoSansThai-Bold.ttf"
FONT_LATIN_REGULAR = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
FONT_LATIN_BOLD = "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"
FONT_MONO = "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf"


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("NotoThai", FONT_REGULAR))
    pdfmetrics.registerFont(TTFont("NotoThaiBold", FONT_BOLD))
    pdfmetrics.registerFont(TTFont("NotoLatin", FONT_LATIN_REGULAR))
    pdfmetrics.registerFont(TTFont("NotoLatinBold", FONT_LATIN_BOLD))
    pdfmetrics.registerFont(TTFont("NotoMono", FONT_MONO))
    pdfmetrics.registerFontFamily(
        "NotoThai",
        normal="NotoThai",
        bold="NotoThaiBold",
        italic="NotoThai",
        boldItalic="NotoThaiBold",
    )


def normalize_text(value: str) -> str:
    return (
        value.replace("\u2011", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("\u2192", "->")
        .replace("\u0394", "delta ")
        .replace("\u03b1", "alpha")
        .replace("\u2265", ">=")
        .replace("\u2264", "<=")
        .replace("\u00a0", " ")
    )


def mixed_script_markup(value: str, *, bold: bool = False) -> str:
    """Use explicit fallback fonts so Thai and Latin glyphs both render."""
    parts: list[str] = []
    for run in re.split(r"([\u0e00-\u0e7f]+)", value):
        if not run:
            continue
        is_thai = bool(re.fullmatch(r"[\u0e00-\u0e7f]+", run))
        if is_thai:
            font = "NotoThaiBold" if bold else "NotoThai"
        else:
            font = "NotoLatinBold" if bold else "NotoLatin"
        parts.append(f'<font name="{font}">{html.escape(run)}</font>')
    return "".join(parts)


def inline_markup(value: str) -> str:
    value = normalize_text(value.strip())
    result: list[str] = []
    token_pattern = re.compile(r"(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)")
    cursor = 0
    for match in token_pattern.finditer(value):
        result.append(mixed_script_markup(value[cursor : match.start()]))
        token = match.group(0)
        if token.startswith("`"):
            result.append(
                f'<font name="NotoMono" color="#24364B">{html.escape(token[1:-1])}</font>'
            )
        elif token.startswith("**"):
            result.append(mixed_script_markup(token[2:-2], bold=True))
        else:
            result.append(f"<i>{mixed_script_markup(token[1:-1])}</i>")
        cursor = match.end()
    result.append(mixed_script_markup(value[cursor:]))
    return "".join(result)


def parse_table(lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for index, line in enumerate(lines):
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if index == 1 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        rows.append(cells)
    return rows


def column_widths(rows: list[list[str]], width: float) -> list[float]:
    columns = len(rows[0])
    if columns >= 8:
        weights = [1.05 if i < 2 else 1.0 for i in range(columns)]
    else:
        weights = []
        for column in range(columns):
            longest = max(len(normalize_text(row[column])) for row in rows)
            weights.append(min(3.0, max(0.8, longest / 16)))
    scale = width / sum(weights)
    return [item * scale for item in weights]


def build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    common = dict(
        fontName="NotoThai",
        textColor=colors.HexColor("#182536"),
        wordWrap="CJK",
        splitLongWords=True,
    )
    return {
        "title": ParagraphStyle(
            "TitleThai",
            parent=base["Title"],
            fontName="NotoThaiBold",
            fontSize=23,
            leading=29,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#12345B"),
            spaceAfter=10,
            wordWrap="CJK",
        ),
        "h2": ParagraphStyle(
            "Heading2Thai",
            parent=base["Heading2"],
            fontName="NotoThaiBold",
            fontSize=15,
            leading=20,
            textColor=colors.HexColor("#0B5D66"),
            spaceBefore=9,
            spaceAfter=5,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "h3": ParagraphStyle(
            "Heading3Thai",
            parent=base["Heading3"],
            fontName="NotoThaiBold",
            fontSize=11.5,
            leading=15,
            textColor=colors.HexColor("#12345B"),
            spaceBefore=6,
            spaceAfter=3,
            keepWithNext=True,
            wordWrap="CJK",
        ),
        "body": ParagraphStyle(
            "BodyThai",
            parent=base["BodyText"],
            fontSize=8.4,
            leading=12.2,
            spaceAfter=4,
            **common,
        ),
        "meta": ParagraphStyle(
            "MetaThai",
            parent=base["BodyText"],
            fontName="NotoThai",
            fontSize=8.2,
            leading=11.5,
            textColor=colors.HexColor("#4A6075"),
            spaceAfter=2,
            wordWrap="CJK",
        ),
        "bullet": ParagraphStyle(
            "BulletThai",
            parent=base["BodyText"],
            fontSize=8.4,
            leading=12.2,
            leftIndent=12,
            firstLineIndent=-9,
            spaceAfter=2.5,
            bulletIndent=2,
            **common,
        ),
        "quote": ParagraphStyle(
            "QuoteThai",
            parent=base["BodyText"],
            fontSize=8.2,
            leading=11.5,
            leftIndent=12,
            rightIndent=12,
            borderColor=colors.HexColor("#70A7A9"),
            borderWidth=0.8,
            borderPadding=6,
            backColor=colors.HexColor("#F1F7F7"),
            textColor=colors.HexColor("#29434E"),
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "table": ParagraphStyle(
            "TableThai",
            parent=base["BodyText"],
            fontName="NotoThai",
            fontSize=6.3,
            leading=8.3,
            textColor=colors.HexColor("#182536"),
            wordWrap="CJK",
            splitLongWords=True,
        ),
        "table_small": ParagraphStyle(
            "TableSmallThai",
            parent=base["BodyText"],
            fontName="NotoThai",
            fontSize=5.45,
            leading=7.1,
            textColor=colors.HexColor("#182536"),
            wordWrap="CJK",
            splitLongWords=True,
        ),
        "table_header": ParagraphStyle(
            "TableHeaderThai",
            parent=base["BodyText"],
            fontName="NotoThaiBold",
            fontSize=6.0,
            leading=7.8,
            textColor=colors.white,
            wordWrap="CJK",
            splitLongWords=True,
        ),
    }


def make_table(rows: list[list[str]], available_width: float, styles: dict[str, ParagraphStyle]) -> Table:
    columns = len(rows[0])
    body_style = styles["table_small"] if columns >= 8 else styles["table"]
    converted: list[list[Paragraph]] = []
    for row_index, row in enumerate(rows):
        style = styles["table_header"] if row_index == 0 else body_style
        converted.append([Paragraph(inline_markup(cell), style) for cell in row])
    table = Table(
        converted,
        colWidths=column_widths(rows, available_width),
        repeatRows=1,
        splitByRow=1,
        hAlign="LEFT",
    )
    commands: list[tuple] = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#174C63")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.28, colors.HexColor("#A8BAC5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]
    for row_index in range(1, len(rows)):
        if row_index % 2 == 0:
            commands.append(
                ("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#F1F5F7"))
            )
    table.setStyle(TableStyle(commands))
    return table


def render_markdown(markdown_path: Path, output_path: Path) -> None:
    register_fonts()
    styles = build_styles()
    page_size = landscape(A4)
    left_margin = 12 * mm
    right_margin = 12 * mm
    top_margin = 15 * mm
    bottom_margin = 14 * mm
    available_width = page_size[0] - left_margin - right_margin
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=page_size,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin,
        title="รายงานฉบับสมบูรณ์ Qwen3-VL CXR Benchmark",
        author="HPCP VLM Experiment",
        subject="Development/Pilot benchmark report",
    )

    story = []
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    index = 0
    title_seen = False
    while index < len(lines):
        line = lines[index].rstrip()
        if not line:
            story.append(Spacer(1, 2.5))
            index += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].startswith("|"):
                table_lines.append(lines[index])
                index += 1
            rows = parse_table(table_lines)
            if rows:
                story.append(make_table(rows, available_width, styles))
                story.append(Spacer(1, 5))
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline_markup(line[2:]), styles["title"]))
            story.append(Spacer(1, 2))
            title_seen = True
        elif line.startswith("## "):
            if title_seen and line.startswith("## 1."):
                story.append(PageBreak())
            else:
                story.append(CondPageBreak(25 * mm))
            story.append(Paragraph(inline_markup(line[3:]), styles["h2"]))
        elif line.startswith("### "):
            story.append(CondPageBreak(18 * mm))
            story.append(Paragraph(inline_markup(line[4:]), styles["h3"]))
        elif line.startswith("> "):
            story.append(Paragraph(inline_markup(line[2:]), styles["quote"]))
        elif re.match(r"^[-*] ", line):
            item_lines = [line[2:].strip()]
            index += 1
            while index < len(lines) and lines[index].startswith(("  ", "\t")):
                item_lines.append(lines[index].strip())
                index += 1
            story.append(
                Paragraph(f"- {inline_markup(' '.join(item_lines))}", styles["bullet"])
            )
            continue
        elif re.match(r"^\d+\. ", line):
            match = re.match(r"^(\d+)\.\s+(.*)", line)
            assert match
            item_lines = [match.group(2).strip()]
            index += 1
            while index < len(lines) and lines[index].startswith(("  ", "\t")):
                item_lines.append(lines[index].strip())
                index += 1
            story.append(
                Paragraph(
                    f"{match.group(1)}. {inline_markup(' '.join(item_lines))}",
                    styles["bullet"],
                )
            )
            continue
        elif line.startswith("**") and ":**" in line:
            story.append(Paragraph(inline_markup(line), styles["meta"]))
        else:
            paragraph_lines = [line]
            index += 1
            while index < len(lines):
                next_line = lines[index].rstrip()
                if (
                    not next_line
                    or next_line.startswith("#")
                    or next_line.startswith("|")
                    or next_line.startswith("> ")
                    or re.match(r"^[-*] ", next_line)
                    or re.match(r"^\d+\. ", next_line)
                ):
                    break
                paragraph_lines.append(next_line)
                index += 1
            story.append(Paragraph(inline_markup(" ".join(paragraph_lines)), styles["body"]))
            continue
        index += 1

    def decorate_page(canvas, doc) -> None:
        canvas.saveState()
        width, height = page_size
        canvas.setFillColor(colors.HexColor("#12345B"))
        canvas.rect(0, height - 7 * mm, width, 7 * mm, fill=1, stroke=0)
        canvas.setFillColor(colors.HexColor("#526777"))
        canvas.setFont("NotoLatin", 6.8)
        canvas.drawString(left_margin, 6 * mm, "Qwen3-VL CXR Development/Pilot Report")
        canvas.drawRightString(width - right_margin, 6 * mm, str(doc.page))
        canvas.restoreState()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    document.build(story, onFirstPage=decorate_page, onLaterPages=decorate_page)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("markdown", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render_markdown(args.markdown.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
