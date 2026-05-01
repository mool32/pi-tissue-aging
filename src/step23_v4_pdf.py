"""
Step 23: Build v4 PDF from pi_tissue_paper_v4.md.

Uses existing figures from step19 + step22 (dividing vs nondividing).
Lightweight — no new figure generation beyond what already exists.
"""
import re
from pathlib import Path
from fpdf import FPDF

BASE = Path("/Users/teo/Desktop/research/pi_tissue_paper")
MD_PATH = BASE / "manuscript" / "pi_tissue_paper_v4.md"
FIG_DIR = BASE / "manuscript" / "figures"
STEP22_FIG = BASE / "results" / "step22_dividing" / "dividing_vs_nondividing.png"
OUT_PDF = BASE / "paper" / "pi_tissue_paper_v4.pdf"

# Figure mapping: section title substring → (image path, caption)
FIGURE_MAP = {
    "Transcriptomic noise accumulates within tissue identity": [
        (FIG_DIR / "fig1_core.png",
         "Figure 1. Variance decomposition in GTEx v8. "
         "pi_tissue = 0.764 to 0.733, pi_donor stable, pi_residual grows "
         "from 0.168 to 0.194. The residual absorbs nearly all age-related "
         "change; tissue identity and donor-level systemic factors are "
         "essentially preserved."),
    ],
    "residual growth is partly compositional, partly cell-intrinsic": [
        (FIG_DIR / "fig2_v3_crossplatform.png",
         "Figure 2. Cross-platform single-cell validation. "
         "Smart-seq2 shows 29% age-related gene-detection decline; "
         "10x Chromium only 10%. On 10x, cross-balanced analysis shows "
         "all 4 cell types with negative Delta_pi (mean = -0.07), a real "
         "but modest cell-intrinsic component."),
    ],
    "Per-tissue noise accumulation is highly heterogeneous": [
        (STEP22_FIG,
         "Figure 3. Per-tissue noise accumulation across 15 GTEx tissues. "
         "Median Delta_var (old - young) with 95% bootstrap CI. Tissues "
         "colored by a priori turnover class (dividing / intermediate / "
         "post-mitotic). Whole blood (+0.079), esophagus (+0.062) and "
         "heart LV (+0.121) are highest; skin and tibial artery near zero. "
         "A simple dividing-vs-post-mitotic dichotomy does not explain the "
         "pattern (Mann-Whitney p = 0.53)."),
    ],
    "Chromatin remodeling genes lose tissue specificity": [
        (FIG_DIR / "fig3_chromatin.png",
         "Figure 4. Chromatin gene cascade. Chromatin remodeling genes "
         "(n=30) lose tissue specificity 2.5-fold faster than "
         "expression-matched controls (p = 0.009). Hierarchy: chromatin > "
         "TFs > targets > housekeeping."),
    ],
    "Caloric restriction partially reverses noise accumulation": [
        (FIG_DIR / "fig4_v3_bootstrap_cr.png",
         "Figure 5. Caloric restriction rescues pi in rat bone marrow via "
         "noise reduction (87% rescue, 95% CI 82-91%). V_residual decreases "
         "in 100% of bootstrap iterations; V_tissue is essentially "
         "unchanged. CR acts as a noise filter, not a structure restorer."),
    ],
    "Cross-species scaling: erosion rate scales inversely": [
        (FIG_DIR / "fig5_v3_scaling.png",
         "Figure 6. Cross-species scaling of pi_tissue erosion rate. "
         "Four mammals spanning ~30-fold in lifespan follow |dpi/dt| "
         "proportional to L^alpha with alpha = -1.02 +/- 0.24 "
         "(R^2 = 0.90, Spearman rho = -1.0)."),
    ],
}


def _sanitize(text):
    """Use cp1252 encoding which supports em-dash, en-dash natively.
    Non-cp1252 chars (Greek letters, mathematical symbols) replaced explicitly.
    """
    r = {
        # Minus sign (U+2212) — cp1252 doesn't have it; use hyphen
        "\u2212": "-",
        # Arrows and mathematical symbols
        "\u2192": "->", "\u21D2": "=>",
        "\u2248": "~", "\u2265": ">=", "\u2264": "<=",
        # Greek letters (not in cp1252)
        "\u03c0": "pi", "\u0394": "Delta", "\u03b1": "alpha", "\u03c1": "rho",
        "\u03b2": "beta", "\u03b3": "gamma", "\u03c4": "tau",
        "\u03ba": "kappa", "\u03bc": "mu", "\u03c3": "sigma",
        "\u03C3": "sigma", "\u03A3": "Sigma",
        # Subscripts / superscripts (some not in cp1252)
        "\u2082": "_2", "\u2083": "_3",
        # Checkmarks
        "\u2713": "[OK]", "\u2717": "[X]",
        # cp1252 SUPPORTS: \u2014 em-dash, \u2013 en-dash, \u00b1, \u00d7,
        # \u00b2, \u00b3, \u00b0, \u00e9 — leave them alone for native render.
    }
    for a, b in r.items():
        text = text.replace(a, b)

    # Try cp1252 first (supports em-dash, en-dash); fall back to '?' for rest
    try:
        text = text.encode("cp1252", errors="replace").decode("cp1252")
    except Exception:
        text = text.encode("latin-1", errors="replace").decode("latin-1")

    # Break very long non-whitespace tokens so fpdf can wrap them
    def _break(m):
        s = m.group(0)
        if len(s) <= 40:
            return s
        return " ".join(s[i:i+35] for i in range(0, len(s), 35))
    text = re.sub(r"\S{40,}", _break, text)
    return text


class PaperPDF(FPDF):
    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.set_auto_page_break(True, margin=18)
        # Use cp1252 which supports em-dash, en-dash, smart quotes
        self.core_fonts_encoding = "cp1252"

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 5, "Spiro 2026 - pi_tissue v4", align="L")
            self.ln(8)

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"{self.page_no()}", align="C")

    def title_block(self):
        self.ln(18)
        self.set_text_color(0, 0, 0)
        self.set_font("Helvetica", "B", 14)
        title = ("Transcriptomic noise accumulates within tissue identity "
                 "across human aging: a systemic signature distinct from "
                 "cell-composition drift")
        self.multi_cell(0, 7, _sanitize(title), align="C")
        self.ln(6)
        self.set_font("Helvetica", "", 11)
        self.cell(0, 6, "Theodor Spiro", align="C")
        self.ln(5)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(80, 80, 80)
        self.cell(0, 5, "Vaika, Inc., 1933 Sweet Rd., East Aurora, NY 14052-3016, USA", align="C")
        self.ln(4)
        self.cell(0, 5, "tspiro@vaika.org", align="C")
        self.ln(5)
        self.cell(0, 5, "Draft v4 -- 2026-04-19", align="C")
        self.ln(10)

    def section(self, title, level=1):
        title = _sanitize(title)
        self.set_text_color(0, 0, 0)
        if level == 1:
            self.ln(4)
            self.set_font("Helvetica", "B", 13)
            self.cell(0, 7, title)
            self.ln(7)
            self.set_draw_color(0, 100, 0)
            self.set_line_width(0.4)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(3)
        elif level == 2:
            self.ln(3)
            self.set_font("Helvetica", "B", 11)
            self.cell(0, 6, title)
            self.ln(7)
        elif level == 3:
            self.ln(2)
            self.set_font("Helvetica", "BI", 10)
            self.cell(0, 5, title)
            self.ln(6)

    def body(self, text):
        text = _sanitize(text)
        self.set_font("Helvetica", "", 9.2)
        self.set_text_color(30, 30, 30)
        self.set_x(self.l_margin)
        try:
            self.multi_cell(0, 4.6, text, align="J")
        except Exception as e:
            print(f"[warn] body failed: {e}; first 80: {text[:80]}")
        self.ln(1)

    def table(self, headers, rows):
        if not headers:
            return
        self.set_font("Helvetica", "B", 8)
        content_w = self.w - self.l_margin - self.r_margin
        col_w = content_w / len(headers)
        self.set_fill_color(225, 240, 225)
        for h in headers:
            self.cell(col_w, 5.5, _sanitize(h), border=1, fill=True, align="C")
        self.ln()
        self.set_font("Helvetica", "", 8)
        for i, row in enumerate(rows):
            if i % 2 == 1:
                self.set_fill_color(248, 248, 248)
            else:
                self.set_fill_color(255, 255, 255)
            for val in row:
                self.cell(col_w, 4.8, _sanitize(str(val)), border=1, fill=True, align="C")
            self.ln()
        self.ln(2)

    def figure(self, img_path, caption="", width=165):
        img_path = Path(img_path)
        if not img_path.exists():
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(200, 0, 0)
            self.cell(0, 5, f"[Figure missing: {img_path.name}]")
            self.ln(5)
            return
        if self.get_y() + 85 > self.h - 30:
            self.add_page()
        x = (self.w - width) / 2
        self.image(str(img_path), x=x, w=width)
        self.ln(2)
        if caption:
            self.set_font("Helvetica", "I", 7.5)
            self.set_text_color(60, 60, 60)
            self.multi_cell(0, 3.4, _sanitize(caption), align="J")
            self.ln(3)


def parse_md(md_path):
    text = md_path.read_text(encoding="utf-8")
    text = text.replace("**", "")
    sections = []
    cur_level, cur_title, cur_body = 0, None, []
    for line in text.split("\n"):
        if line.startswith("### "):
            if cur_title is not None:
                sections.append((cur_level, cur_title, "\n".join(cur_body).strip()))
            cur_level = 3; cur_title = line[4:].strip(); cur_body = []
        elif line.startswith("## "):
            if cur_title is not None:
                sections.append((cur_level, cur_title, "\n".join(cur_body).strip()))
            cur_level = 2; cur_title = line[3:].strip(); cur_body = []
        elif line.startswith("# "):
            if cur_title is not None:
                sections.append((cur_level, cur_title, "\n".join(cur_body).strip()))
            cur_level = 1; cur_title = line[2:].strip(); cur_body = []
        elif line.startswith("---"):
            continue
        else:
            cur_body.append(line)
    if cur_title:
        sections.append((cur_level, cur_title, "\n".join(cur_body).strip()))
    return sections


def is_table_block(para):
    lines = para.split("\n")
    if len(lines) < 2:
        return False
    if "|" not in lines[0] or "---" not in lines[1]:
        return False
    return lines[0].count("|") >= 3


def parse_table(para):
    lines = para.split("\n")
    headers = [c.strip() for c in lines[0].split("|") if c.strip()]
    rows = []
    for tl in lines[2:]:
        if "|" in tl:
            row = [c.strip() for c in tl.split("|") if c.strip()]
            if row:
                rows.append(row)
    return headers, rows


def build():
    sections = parse_md(MD_PATH)
    print(f"Parsed {len(sections)} sections")

    pdf = PaperPDF()
    pdf.add_page()
    pdf.title_block()

    for level, title, body in sections:
        if level == 1:
            continue  # skip top-level title (rendered in title block)
        if title in ("Authors",):
            continue

        major = ("Abstract", "1. Introduction", "2. Data sources", "3. Results",
                 "4. Discussion", "5. Methods", "Figure Legends",
                 "References", "Acknowledgements")
        if level == 2 and title in major and title != "Abstract":
            pdf.add_page()

        pdf.section(title, level)

        if body.strip():
            paras = re.split(r"\n\s*\n", body.strip())
            for para in paras:
                para = para.strip()
                if not para:
                    continue
                if is_table_block(para):
                    headers, rows = parse_table(para)
                    if headers and rows:
                        pdf.table(headers, rows)
                else:
                    # bullet list
                    if para.startswith("- ") or para.startswith("* "):
                        for ln in para.split("\n"):
                            if ln.startswith("- ") or ln.startswith("* "):
                                pdf.set_font("Helvetica", "", 9)
                                pdf.set_text_color(30, 30, 30)
                                pdf.set_x(pdf.l_margin)
                                try:
                                    pdf.multi_cell(0, 4.3, _sanitize("  - " + ln[2:].strip()), align="L")
                                except Exception:
                                    pass
                            elif ln.strip():
                                try:
                                    pdf.multi_cell(0, 4.3, _sanitize(ln.strip()), align="L")
                                except Exception:
                                    pass
                        pdf.ln(1)
                    elif para.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.")):
                        # numbered list
                        pdf.set_font("Helvetica", "", 9)
                        pdf.set_text_color(30, 30, 30)
                        for ln in para.split("\n"):
                            if ln.strip():
                                pdf.set_x(pdf.l_margin)
                                try:
                                    pdf.multi_cell(0, 4.3, _sanitize(ln.strip()), align="L")
                                except Exception:
                                    pass
                        pdf.ln(1)
                    else:
                        txt = " ".join(l.strip() for l in para.split("\n"))
                        pdf.body(txt)

        # Insert figures after matching section
        for key, figs in FIGURE_MAP.items():
            if key in title:
                for fig_path, cap in figs:
                    pdf.figure(fig_path, cap)

    pdf.output(str(OUT_PDF))
    size_kb = OUT_PDF.stat().st_size / 1024
    print(f"Saved {OUT_PDF} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    build()
