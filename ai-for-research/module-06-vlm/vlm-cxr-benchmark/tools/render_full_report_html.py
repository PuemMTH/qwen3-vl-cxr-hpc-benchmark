#!/usr/bin/env python3
"""Render the verified Markdown experiment report as a self-contained HTML file."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import markdown


STYLES = r"""
:root {
  --ink: #172536;
  --muted: #5e6e7e;
  --navy: #12385c;
  --teal: #0b6a70;
  --line: #d6e0e7;
  --paper: #ffffff;
  --wash: #f3f7f9;
  --warn-bg: #fff8e7;
  --warn-line: #d99b22;
  --shadow: 0 16px 50px rgba(18, 56, 92, 0.09);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  color: var(--ink);
  background: #eaf0f3;
  font-family: "Noto Sans Thai", "Noto Sans", Tahoma, Arial, sans-serif;
  font-size: 16px;
  line-height: 1.72;
}
a { color: var(--teal); text-decoration-thickness: 1px; text-underline-offset: 3px; }
.topbar { height: 8px; background: var(--navy); }
.hero {
  color: white;
  background: var(--navy);
  padding: 52px max(24px, calc((100vw - 1320px) / 2)) 42px;
}
.eyebrow {
  display: inline-flex;
  gap: 8px;
  align-items: center;
  margin-bottom: 16px;
  padding: 6px 10px;
  border: 1px solid rgba(255,255,255,.3);
  border-radius: 4px;
  color: #dcecf2;
  font-size: .78rem;
  font-weight: 700;
  letter-spacing: .08em;
}
.hero h1 {
  max-width: 1040px;
  margin: 0;
  font-size: clamp(2rem, 4.2vw, 3.8rem);
  line-height: 1.18;
  letter-spacing: -.025em;
}
.hero-note { max-width: 900px; margin: 18px 0 0; color: #d6e3ea; font-size: 1.05rem; }
.metrics {
  display: grid;
  grid-template-columns: repeat(5, minmax(150px, 1fr));
  gap: 1px;
  max-width: 1320px;
  margin: 0 auto;
  background: #cddce4;
  border: 1px solid #cddce4;
}
.metric { min-height: 116px; padding: 22px 20px; background: white; }
.metric strong { display: block; color: var(--navy); font-size: 1.65rem; line-height: 1.15; }
.metric span { display: block; margin-top: 8px; color: var(--muted); font-size: .83rem; line-height: 1.35; }
.shell {
  display: grid;
  grid-template-columns: 260px minmax(0, 1fr);
  gap: 34px;
  width: min(1320px, calc(100% - 40px));
  margin: 34px auto 70px;
}
.toc {
  align-self: start;
  position: sticky;
  top: 24px;
  max-height: calc(100vh - 48px);
  overflow: auto;
  padding: 20px;
  border: 1px solid var(--line);
  background: rgba(255,255,255,.94);
  box-shadow: var(--shadow);
}
.toc-title { margin: 0 0 10px; color: var(--navy); font-size: .78rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
.toc ul { margin: 0; padding: 0; list-style: none; }
.toc li { margin: 0; }
.toc li li a { padding-left: 14px; color: #53687a; font-size: .82rem; }
.toc a { display: block; padding: 5px 0; color: #244760; font-size: .88rem; line-height: 1.35; text-decoration: none; }
.toc a:hover, .toc a.active { color: var(--teal); font-weight: 700; }
.report {
  min-width: 0;
  padding: clamp(24px, 5vw, 66px);
  border: 1px solid var(--line);
  background: var(--paper);
  box-shadow: var(--shadow);
}
.scope {
  margin: 0 0 40px;
  padding: 20px 22px;
  border-left: 5px solid var(--warn-line);
  background: var(--warn-bg);
}
.scope strong { display: block; color: #76510b; }
.scope p { margin: 4px 0 0; color: #664f25; }
.report h2, .report h3 { scroll-margin-top: 22px; color: var(--navy); line-height: 1.3; }
.report h2 { margin: 52px 0 18px; padding-top: 10px; border-top: 2px solid var(--navy); font-size: 1.75rem; }
.report h3 { margin: 34px 0 12px; font-size: 1.28rem; }
.report p { margin: 0 0 17px; }
.report ul, .report ol { margin: 0 0 22px; padding-left: 1.45rem; }
.report li { margin: 6px 0; padding-left: .25rem; }
.report code {
  overflow-wrap: anywhere;
  padding: .08em .32em;
  border: 1px solid #d8e2e8;
  border-radius: 3px;
  background: #edf3f6;
  color: #173a58;
  font-family: "Noto Sans Mono", "DejaVu Sans Mono", monospace;
  font-size: .88em;
}
.table-wrap { width: 100%; margin: 20px 0 28px; overflow-x: auto; border: 1px solid #b9cbd5; }
table { width: 100%; min-width: 720px; border-collapse: collapse; font-size: .82rem; line-height: 1.4; font-variant-numeric: tabular-nums; }
th { padding: 10px 11px; border-right: 1px solid rgba(255,255,255,.3); color: white; background: #174f66; text-align: left; vertical-align: bottom; }
td { padding: 9px 11px; border-top: 1px solid #c5d3dc; border-right: 1px solid #d5e0e6; vertical-align: top; }
tbody tr:nth-child(even) { background: var(--wash); }
blockquote { margin: 24px 0; padding: 15px 20px; border-left: 4px solid var(--teal); background: #eef7f7; color: #294951; }
.footer { padding: 30px 20px 44px; color: #6a7986; text-align: center; font-size: .82rem; }
.mobile-toc { display: none; }
@media (max-width: 1040px) {
  .metrics { grid-template-columns: repeat(2, 1fr); }
  .metrics .metric:last-child { grid-column: 1 / -1; }
  .shell { display: block; width: min(920px, calc(100% - 24px)); }
  .toc { display: none; }
  .mobile-toc { display: block; margin: 0 0 20px; padding: 14px 18px; border: 1px solid var(--line); background: white; }
  .mobile-toc summary { cursor: pointer; color: var(--navy); font-weight: 800; }
  .mobile-toc .toc { display: block; position: static; max-height: none; margin-top: 12px; padding: 0; border: 0; box-shadow: none; }
}
@media (max-width: 620px) {
  body { font-size: 15px; }
  .hero { padding: 36px 20px 30px; }
  .metrics { grid-template-columns: 1fr; }
  .metrics .metric:last-child { grid-column: auto; }
  .metric { min-height: 0; }
  .report { padding: 24px 18px; }
  .report h2 { font-size: 1.45rem; }
}
@media print {
  body { background: white; font-size: 10pt; }
  .hero { padding: 22mm 16mm 14mm; print-color-adjust: exact; }
  .hero h1 { font-size: 27pt; }
  .metrics { grid-template-columns: repeat(5, 1fr); print-color-adjust: exact; }
  .metric { min-height: 0; padding: 12px; }
  .metric strong { font-size: 14pt; }
  .shell { display: block; width: auto; margin: 0; }
  .toc, .mobile-toc { display: none !important; }
  .report { border: 0; box-shadow: none; padding: 12mm 16mm; }
  .report h2, .report h3 { break-after: avoid; }
  .table-wrap { overflow: visible; break-inside: avoid; }
  table { min-width: 0; font-size: 7pt; }
  a { color: inherit; text-decoration: none; }
}
"""


SCRIPT = r"""
const links = [...document.querySelectorAll('.toc a[href^="#"]')];
const sections = links.map(a => document.querySelector(a.getAttribute('href'))).filter(Boolean);
const observer = new IntersectionObserver(entries => {
  const visible = entries.filter(e => e.isIntersecting).sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
  if (!visible) return;
  links.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + visible.target.id));
}, { rootMargin: '-12% 0px -78% 0px' });
sections.forEach(section => observer.observe(section));
"""


def build_html(markdown_path: Path) -> str:
    source = markdown_path.read_text(encoding="utf-8")
    title_match = re.match(r"^#\s+(.+)$", source, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "VLM CXR Experiment Report"
    converter = markdown.Markdown(
        extensions=["tables", "toc", "sane_lists"],
        extension_configs={"toc": {"permalink": False, "toc_depth": "2-3"}},
    )
    body = converter.convert(source)
    body = re.sub(r"^<h1[^>]*>.*?</h1>\s*", "", body, count=1, flags=re.DOTALL)
    body = re.sub(r"(<table>.*?</table>)", r'<div class="table-wrap">\1</div>', body, flags=re.DOTALL)
    toc = converter.toc
    safe_title = html.escape(title)
    return f"""<!doctype html>
<html lang="th">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <meta name="description" content="Development/Pilot report for Qwen3-VL chest X-ray experiments on RTX 5070 and LANTA A100">
  <title>{safe_title}</title>
  <style>{STYLES}</style>
</head>
<body>
  <div class="topbar"></div>
  <header class="hero">
    <div class="eyebrow">DEVELOPMENT / PILOT · TEST LOCKED</div>
    <h1>{safe_title}</h1>
    <p class="hero-note">ผลการทดลองที่รันจริงบน RTX 5070 และ LANTA A100 พร้อม metrics, confidence intervals, resource use และขอบเขตหลักฐาน</p>
  </header>
  <section class="metrics" aria-label="ผลลัพธ์สำคัญ">
    <div class="metric"><strong>4B BF16</strong><span>Development comparator ที่ถูกเลือก</span></div>
    <div class="metric"><strong>0.6867</strong><span>patient-level balanced accuracy</span></div>
    <div class="metric"><strong>41,614</strong><span>images/hour ที่ batch 8 บน A100</span></div>
    <div class="metric"><strong>3.914×</strong><span>speedup ที่ 4 GPUs</span></div>
    <div class="metric"><strong>9.3200</strong><span>allocated GPU-hours รวม</span></div>
  </section>
  <div class="shell">
    <aside class="toc" aria-label="สารบัญ"><p class="toc-title">สารบัญ</p>{toc}</aside>
    <div>
      <details class="mobile-toc"><summary>เปิดสารบัญ</summary><nav class="toc">{toc}</nav></details>
      <main class="report">
        <section class="scope">
          <strong>ขอบเขตรายงาน</strong>
          <p>การทดลอง Development/Pilot รันเสร็จแล้ว แต่ยังไม่ได้เปิด official V3 confirmatory Test รายงานนี้จึงไม่ใช่ผลยืนยันทางคลินิกหรือผล confirmatory ขั้นสุดท้าย</p>
        </section>
        {body}
      </main>
    </div>
  </div>
  <footer class="footer">Qwen3-VL CXR Development/Pilot · Generated from the verified experiment report</footer>
  <script>{SCRIPT}</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("markdown", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(args.markdown.resolve()), encoding="utf-8")


if __name__ == "__main__":
    main()
