"""
Step 21: Generate new v3 figures + compile PDF manuscript.
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF

BASE = Path("/Users/teo/Desktop/research/pi_tissue_paper")
FIG_DIR = BASE / "manuscript" / "figures"
V_DIR = BASE / "results" / "step20_verification"
OUT_PDF = BASE / "manuscript" / "pi_tissue_paper_v3.pdf"

# ── Style ──
COLORS = {
    "tissue": "#2ca02c", "donor": "#1f77b4", "residual": "#999999",
    "red": "#d62728", "orange": "#ff7f0e", "purple": "#9467bd",
    "teal": "#17becf", "blue": "#1f77b4", "green": "#2ca02c",
    "smartseq2": "#d62728", "chromium10x": "#1f77b4",
}
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})

def _save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# ══════════════════════════════════════════════════════════════
# FIGURE GENERATION
# ══════════════════════════════════════════════════════════════

def gen_fig2_crossplatform():
    """Fig 2: Cross-platform SC validation."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1, 1.3, 1]})

    # Panel A: Gene detection QC
    ax = axes[0]
    platforms = ["Smart-seq2\n(FACS)", "10x Chromium\n(Droplet)"]
    young_det = [2824, 1845]
    old_det = [2002, 1657]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w/2, young_det, w, label="Young (3m)", color=COLORS["blue"], alpha=0.8)
    ax.bar(x + w/2, old_det, w, label="Old (18-24m)", color=COLORS["red"], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(platforms, fontsize=8)
    ax.set_ylabel("Genes / cell")
    ax.set_title("A  Gene detection by platform", fontsize=10, fontweight="bold", loc="left")
    ax.legend(fontsize=7)
    # Annotate decline %
    for i, (y, o) in enumerate(zip(young_det, old_det)):
        pct = (y - o) / y * 100
        ax.annotate(f"-{pct:.0f}%", (i + w/2, o), ha="center", va="bottom", fontsize=7,
                    color=COLORS["red"], fontweight="bold")

    # Panel B: 10x cross-balanced Δπ
    ax = axes[1]
    df = pd.read_csv(V_DIR / "v4_tms_droplet_cross_balanced.csv")
    cts = []
    deltas = []
    for ct in df["cell_type"].unique():
        y = df[(df["cell_type"] == ct) & (df["age_group"] == "young")]
        o = df[(df["cell_type"] == ct) & (df["age_group"] == "old")]
        if len(y) > 0 and len(o) > 0:
            cts.append(ct.replace("cell", "").strip())
            deltas.append(o["pi_tissue"].iloc[0] - y["pi_tissue"].iloc[0])

    bars = ax.barh(range(len(cts)), deltas, color=[COLORS["red"] if d < 0 else COLORS["green"] for d in deltas],
                   alpha=0.8, edgecolor="none")
    ax.set_yticks(range(len(cts)))
    ax.set_yticklabels(cts, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel(r"$\Delta\pi_{tissue}$ (old - young)")
    ax.set_title("B  10x Chromium (cross-balanced)", fontsize=10, fontweight="bold", loc="left")
    for i, d in enumerate(deltas):
        ax.text(d - 0.005 if d < 0 else d + 0.005, i, f"{d:+.03f}",
                ha="right" if d < 0 else "left", va="center", fontsize=7)
    ax.set_xlim(-0.15, 0.03)

    # Panel C: FACS vs 10x comparison (inset-style)
    ax = axes[2]
    facs_delta = -0.01  # mean from balanced FACS
    tenx_delta = np.mean(deltas)
    bars = ax.bar(["Smart-seq2\n(FACS)", "10x Chromium\n(Droplet)"],
                  [facs_delta, tenx_delta],
                  color=[COLORS["smartseq2"], COLORS["chromium10x"]], alpha=0.8, width=0.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel(r"Mean $\Delta\pi_{tissue}$")
    ax.set_title("C  Platform comparison", fontsize=10, fontweight="bold", loc="left")
    for bar, val in zip(bars, [facs_delta, tenx_delta]):
        ax.text(bar.get_x() + bar.get_width()/2, val, f"{val:+.03f}",
                ha="center", va="top" if val < 0 else "bottom", fontsize=8, fontweight="bold")

    fig.tight_layout()
    _save(fig, "fig2_v3_crossplatform")


def gen_fig4_bootstrap_cr():
    """Fig 4: CR with bootstrap CIs."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))

    # Panel A: pi values with CI
    ax = axes[0]
    summary = pd.read_csv(V_DIR / "v3_bootstrap_summary.csv")
    boot = pd.read_csv(V_DIR / "v3_bootstrap_cr.csv")

    conditions = ["young", "old_AL", "old_CR"]
    labels = ["Young\n(5m)", "Old AL\n(27m)", "Old CR\n(27m)"]
    pi_vals = [boot["pi_young"].median(), boot["pi_old_AL"].median(), boot["pi_old_CR"].median()]
    ci_lo = [boot["pi_young"].quantile(0.025), boot["pi_old_AL"].quantile(0.025), boot["pi_old_CR"].quantile(0.025)]
    ci_hi = [boot["pi_young"].quantile(0.975), boot["pi_old_AL"].quantile(0.975), boot["pi_old_CR"].quantile(0.975)]
    errs = [[p - l for p, l in zip(pi_vals, ci_lo)], [h - p for p, h in zip(pi_vals, ci_hi)]]
    colors_bar = [COLORS["green"], COLORS["red"], COLORS["orange"]]

    ax.bar(range(3), pi_vals, yerr=errs, color=colors_bar, alpha=0.8, capsize=5, edgecolor="none")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel(r"$\pi_{tissue}$")
    ax.set_ylim(0.82, 0.90)
    ax.set_title("A  CR rescues 87% [82-91%]", fontsize=10, fontweight="bold", loc="left")
    # Arrow showing rescue
    ax.annotate("", xy=(2, pi_vals[2]), xytext=(1, pi_vals[1]),
                arrowprops=dict(arrowstyle="->", color=COLORS["orange"], lw=2))

    # Panel B: Bootstrap rescue distribution
    ax = axes[1]
    rescue = boot["rescue_pct"].dropna()
    ax.hist(rescue, bins=40, color=COLORS["orange"], alpha=0.7, edgecolor="none")
    ax.axvline(rescue.median(), color="black", lw=2, label=f"Median: {rescue.median():.1f}%")
    ax.axvline(rescue.quantile(0.025), color="black", lw=1, ls="--", label=f"95% CI: [{rescue.quantile(0.025):.1f}, {rescue.quantile(0.975):.1f}]%")
    ax.axvline(rescue.quantile(0.975), color="black", lw=1, ls="--")
    ax.set_xlabel("CR rescue (%)")
    ax.set_ylabel("Bootstrap iterations")
    ax.set_title("B  Bootstrap (n=1000)", fontsize=10, fontweight="bold", loc="left")
    ax.legend(fontsize=7, loc="upper left")

    # Panel C: Mechanism — V_tissue vs V_residual change
    ax = axes[2]
    vt = boot["V_tissue_CR_effect"] * 1000  # scale for readability
    vr = boot["V_residual_CR_effect"] * 1000

    parts = ax.violinplot([vt.values, vr.values], positions=[0, 1], showmedians=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor([COLORS["green"], COLORS["blue"]][i])
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")

    ax.axhline(0, color="black", lw=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"$\Delta V_{tissue}$", r"$\Delta V_{residual}$"], fontsize=9)
    ax.set_ylabel(r"CR effect ($\times 10^{-3}$)")
    ax.set_title("C  Mechanism", fontsize=10, fontweight="bold", loc="left")

    # Annotations
    pct_vr = (vr < 0).mean() * 100
    pct_vt = (vt > 0).mean() * 100
    ax.text(0, vt.median() + 0.01, f"{pct_vt:.0f}% > 0", ha="center", va="bottom", fontsize=7, color=COLORS["green"])
    ax.text(1, vr.median() - 0.01, f"{pct_vr:.0f}% < 0", ha="center", va="top", fontsize=7, color=COLORS["blue"])

    fig.tight_layout()
    _save(fig, "fig4_v3_bootstrap_cr")


def gen_fig5_scaling():
    """Fig 5: Macaque + updated scaling law."""
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))

    # Panel A: Macaque trajectory
    ax = axes[0]
    mac = pd.read_csv(V_DIR / "v1_macaque_batch2_balanced.csv")
    ax.errorbar(mac["age_midpoint"], mac["pi_mean"], yerr=mac["pi_std"],
                fmt="o-", color=COLORS["purple"], capsize=4, markersize=6, lw=2)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel(r"$\pi_{tissue}$")
    ax.set_title("A  Macaque (batch 2, balanced)", fontsize=10, fontweight="bold", loc="left")
    ax.set_ylim(0.70, 0.92)
    # Annotate
    for _, r in mac.iterrows():
        ax.annotate(r["age_group"], (r["age_midpoint"], r["pi_mean"]),
                    textcoords="offset points", xytext=(5, 8), fontsize=7)

    # Panel B: 4-species scaling
    ax = axes[1]
    sp = pd.read_csv(V_DIR / "v_scaling_law_4species.csv")
    for _, r in sp.iterrows():
        ax.scatter(r["lifespan_yr"], abs(r["dpi_dt"]), s=80, zorder=5, color=COLORS["blue"])
        ax.annotate(r["species"], (r["lifespan_yr"], abs(r["dpi_dt"])),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    # Fit line
    from scipy import stats
    log_L = np.log10(sp["lifespan_yr"].values)
    log_rate = np.log10(sp["abs_dpi_dt"].values)
    slope, intercept, r_val, p_val, se = stats.linregress(log_L, log_rate)
    x_fit = np.logspace(np.log10(1.5), np.log10(120), 100)
    y_fit = 10**(slope * np.log10(x_fit) + intercept)
    ax.plot(x_fit, y_fit, "--", color=COLORS["red"], alpha=0.7,
            label=rf"$\alpha$ = {slope:.2f} $\pm$ {se:.2f}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Maximum lifespan (years)")
    ax.set_ylabel(r"|$d\pi/dt$| (per year)")
    ax.set_title(rf"B  Scaling law (R$^2$={r_val**2:.2f})", fontsize=10, fontweight="bold", loc="left")
    ax.legend(fontsize=7)

    # Panel C: Mouse correction
    ax = axes[2]
    mouse = pd.read_csv(V_DIR / "v2_mouse_slopes.csv")
    mouse_pi = pd.read_csv(BASE / "results" / "step16_final" / "mouse_bulk_pi.csv")

    ax.plot(mouse_pi["age"] / 12, mouse_pi["pi"], "o-", color=COLORS["blue"], markersize=4, lw=1.5)
    ax.axvline(3/12, color=COLORS["red"], ls="--", lw=0.8, alpha=0.5)
    ax.text(3/12 + 0.02, 0.60, "Adult\ncutoff", fontsize=7, color=COLORS["red"], va="top")

    # Show adult slope
    adult_row = mouse[mouse["cutoff"] == ">=3m"]
    if len(adult_row) > 0:
        s = adult_row.iloc[0]
        x_line = np.array([3/12, 27/12])
        # dpi_dt is per year, we need intercept
        from scipy.stats import linregress
        adult_pi = mouse_pi[mouse_pi["age"] >= 3]
        sl, ic, _, _, _ = linregress(adult_pi["age"].values / 12, adult_pi["pi"].values)
        ax.plot(x_line, sl * x_line + ic, "--", color=COLORS["red"], lw=2,
                label=f"Adult slope: {sl:+.03f}/yr")

    ax.set_xlabel("Age (years)")
    ax.set_ylabel(r"$\pi_{tissue}$")
    ax.set_title("C  Mouse (bulk, corrected)", fontsize=10, fontweight="bold", loc="left")
    ax.legend(fontsize=7)
    ax.set_ylim(0.40, 0.65)

    fig.tight_layout()
    _save(fig, "fig5_v3_scaling")


def generate_all_figures():
    print("Generating v3 figures...")
    gen_fig2_crossplatform()
    gen_fig4_bootstrap_cr()
    gen_fig5_scaling()
    print("All v3 figures generated.")


# ══════════════════════════════════════════════════════════════
# PDF MANUSCRIPT
# ══════════════════════════════════════════════════════════════

class ManuscriptPDF(FPDF):
    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.set_auto_page_break(True, margin=20)
        # Use built-in fonts only (Helvetica = Arial equivalent)

    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 5, "pi_tissue manuscript v3", align="L")
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def add_title_page(self, title, keywords):
        self.add_page()
        self.ln(40)
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 10, title, align="C")
        self.ln(15)
        self.set_font("Helvetica", "", 11)
        self.cell(0, 8, "[Authors to be completed]", align="C")
        self.ln(20)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(80, 80, 80)
        self.multi_cell(0, 5, f"Keywords: {keywords}", align="C")

    def add_section(self, title, level=1):
        title = _sanitize(title)
        self.set_text_color(0, 0, 0)
        if level == 1:
            self.ln(6)
            self.set_font("Helvetica", "B", 14)
            self.cell(0, 8, title)
            self.ln(10)
            # Draw line under section
            self.set_draw_color(0, 100, 0)
            self.set_line_width(0.5)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(4)
        elif level == 2:
            self.ln(4)
            self.set_font("Helvetica", "B", 11)
            self.cell(0, 7, title)
            self.ln(8)
        elif level == 3:
            self.ln(3)
            self.set_font("Helvetica", "BI", 10)
            self.cell(0, 6, title)
            self.ln(7)

    def add_body(self, text):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(30, 30, 30)
        text = _sanitize(text)
        self.multi_cell(0, 4.5, text, align="J")
        self.ln(2)

    def add_figure(self, img_path, caption="", width=170):
        if not Path(img_path).exists():
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(200, 0, 0)
            self.cell(0, 6, f"[Figure not found: {Path(img_path).name}]")
            self.ln(4)
            return

        # Check if enough space, otherwise new page
        if self.get_y() + 80 > self.h - 30:
            self.add_page()

        x = (self.w - width) / 2
        self.image(str(img_path), x=x, w=width)
        self.ln(3)
        if caption:
            self.set_font("Helvetica", "I", 7.5)
            self.set_text_color(60, 60, 60)
            self.multi_cell(0, 3.5, _sanitize(caption), align="J")
            self.ln(4)

    def add_table(self, headers, rows):
        self.set_font("Helvetica", "B", 8)
        col_w = (self.w - self.l_margin - self.r_margin) / len(headers)
        # Header
        self.set_fill_color(230, 240, 230)
        for h in headers:
            self.cell(col_w, 6, _sanitize(h), border=1, fill=True, align="C")
        self.ln()
        # Rows
        self.set_font("Helvetica", "", 8)
        self.set_text_color(30, 30, 30)
        for i, row in enumerate(rows):
            if i % 2 == 1:
                self.set_fill_color(245, 245, 245)
            else:
                self.set_fill_color(255, 255, 255)
            for val in row:
                self.cell(col_w, 5, _sanitize(str(val)), border=1, fill=True, align="C")
            self.ln()
        self.ln(3)


def _sanitize(text):
    """Replace non-latin1 characters for fpdf2 core fonts."""
    replacements = {
        "\u2014": " -- ", "\u2013": " - ",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u03c0": "pi", "\u0394": "Delta", "\u03b1": "alpha", "\u03c1": "rho",
        "\u03b2": "beta", "\u03b3": "gamma", "\u03c4": "tau",
        "\u2248": "~", "\u00d7": "x", "\u2265": ">=", "\u2264": "<=",
        "\u00b1": "+/-", "\u2192": "->",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Remove any remaining non-latin1 chars
    text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text


def parse_markdown(md_path):
    """Parse v3 markdown into structured sections."""
    text = Path(md_path).read_text(encoding="utf-8")

    # Clean up markdown formatting for plain text rendering
    # Remove markdown bold/italic markers for fpdf (can't handle inline)
    text = text.replace("**", "")
    # Replace em-dashes and other unicode with ASCII equivalents
    text = text.replace("\u2014", " -- ")
    text = text.replace("\u2013", " - ")
    text = text.replace("\u2018", "'")
    text = text.replace("\u2019", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    text = text.replace("\u03c0", "pi")
    text = text.replace("\u0394", "Delta")
    text = text.replace("\u03b1", "alpha")
    text = text.replace("\u03c1", "rho")
    text = text.replace("\u2248", "~")
    text = text.replace("\u00d7", "x")

    sections = []
    current_section = None
    current_level = 0
    current_text = []

    for line in text.split("\n"):
        if line.startswith("### "):
            if current_section is not None:
                sections.append((current_level, current_section, "\n".join(current_text).strip()))
            current_section = line[4:].strip()
            current_level = 3
            current_text = []
        elif line.startswith("## "):
            if current_section is not None:
                sections.append((current_level, current_section, "\n".join(current_text).strip()))
            current_section = line[3:].strip()
            current_level = 2
            current_text = []
        elif line.startswith("# "):
            if current_section is not None:
                sections.append((current_level, current_section, "\n".join(current_text).strip()))
            current_section = line[2:].strip()
            current_level = 1
            current_text = []
        elif line.startswith("---"):
            continue
        else:
            current_text.append(line)

    if current_section:
        sections.append((current_level, current_section, "\n".join(current_text).strip()))

    return sections


def build_pdf():
    print("Building PDF manuscript...")
    md_path = BASE / "manuscript" / "pi_tissue_paper_v3.md"
    sections = parse_markdown(md_path)

    pdf = ManuscriptPDF()

    # Title page
    title = "Tissue identity as a transcriptomic near-invariant:\ncompositional drift, noise accumulation,\nand cross-species scaling"
    keywords = "aging, tissue identity, variance decomposition, GTEx, single-cell RNA-seq, caloric restriction, chromatin, epigenetic drift, cross-species scaling"
    pdf.add_title_page(title, keywords)

    # Figure map: which figure to show after which section
    figure_map = {
        "Tissue identity accounts for approximately 75% of transcriptomic variance and is near-invariant with age": [
            (FIG_DIR / "fig1_core.png", "Figure 1. Tissue identity is the dominant organizational mode and is near-invariant with age. GTEx v8, 263 donors, 6 tissues, 18,000 genes.")
        ],
        "Single-cell validation across two platforms: predominantly compositional decline with a cell-intrinsic component": [
            (FIG_DIR / "fig2_v3_crossplatform.png", "Figure 2. Cross-platform single-cell validation. (A) Gene detection QC: Smart-seq2 shows 29% decline vs 10% for 10x. (B) Cross-balanced 10x analysis: all 4 cell types show negative Delta_pi (mean = -0.07). (C) Platform comparison of mean Delta_pi.")
        ],
        "Chromatin remodeling machinery erodes tissue specificity fastest": [
            (FIG_DIR / "fig3_chromatin.png", "Figure 3. Chromatin remodeling genes erode tissue specificity 2.5x faster than expression-matched controls (p = 0.009).")
        ],
        "Caloric restriction rescues pi through noise reduction, not structure repair": [
            (FIG_DIR / "fig4_v3_bootstrap_cr.png", "Figure 4. CR rescues 87% [82-91%] of pi loss via noise reduction. (A) pi values with bootstrap CIs. (B) Bootstrap rescue distribution (n=1000). (C) Mechanism: V_residual decreases in 100% of bootstraps.")
        ],
        "Cross-species scaling: erosion rate inversely proportional to lifespan": [
            (FIG_DIR / "fig5_v3_scaling.png", "Figure 5. Cross-species scaling with macaque. (A) Macaque trajectory (batch 2, balanced). (B) 4-species scaling law: alpha = -1.02, R2 = 0.90. (C) Mouse bulk trajectory (corrected).")
        ],
        "Cancer nearly abolishes tissue identity": [
            (FIG_DIR / "figS2_pertissue_tcga.png", "Figure S5. Cancer abolishes tissue identity: pi_tumor = 0.016 vs pi_normal ~ 0.73 (45-fold reduction).")
        ],
    }

    # Table data for cross-species
    scaling_table_headers = ["Species", "Lifespan (yr)", "pi_young", "pi_old", "dpi/dt (/yr)"]
    scaling_table_rows = [
        ["Mouse", "2.5", "0.607", "0.488", "-0.060"],
        ["Rat", "3.0", "0.893", "0.842", "-0.028"],
        ["Macaque", "40", "0.860", "0.753", "-0.006"],
        ["Human", "80", "0.764", "0.733", "-0.001"],
    ]

    # 10x table
    tenx_table_headers = ["Cell type", "pi_young", "pi_old", "Delta_pi"]
    tenx_table_rows = [
        ["Macrophage", "0.648", "0.560", "-0.088"],
        ["Endothelial", "0.493", "0.382", "-0.110"],
        ["B cell", "0.342", "0.305", "-0.038"],
        ["T cell", "0.384", "0.331", "-0.053"],
    ]

    # Render sections
    for level, title_text, body in sections:
        if title_text in ("Authors", "[To be completed]"):
            continue

        # Special handling for Abstract
        if title_text == "Abstract":
            pdf.add_page()
            pdf.add_section("Abstract", 1)
            # Render abstract in slightly larger font
            pdf.set_font("Helvetica", "", 9.5)
            pdf.set_text_color(30, 30, 30)
            # Clean body
            body_clean = body.replace("\n\n", "\n").strip()
            lines = body_clean.split("\n")
            abstract_text = " ".join(l.strip() for l in lines if l.strip() and not l.startswith("Keywords"))
            pdf.multi_cell(0, 5, _sanitize(abstract_text), align="J")
            pdf.ln(3)
            # Keywords
            kw_line = [l for l in lines if "Keywords:" in l]
            if kw_line:
                pdf.set_font("Helvetica", "I", 8)
                pdf.multi_cell(0, 4, _sanitize(kw_line[0]), align="L")
            continue

        # Section headers
        if level == 1:
            pdf.add_page()

        if level == 1:
            pdf.add_section(title_text, 1)
        elif level == 2:
            pdf.add_section(title_text, 2)
        elif level == 3:
            pdf.add_section(title_text, 3)

        # Body text
        if body.strip():
            # Split into paragraphs (separated by blank lines)
            paragraphs = re.split(r"\n\s*\n", body.strip())
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                lines = para.split("\n")

                # Check if this is a markdown table
                is_table = (len(lines) >= 2
                            and "|" in lines[0]
                            and len(lines) >= 2
                            and "---" in lines[1]
                            and lines[0].count("|") >= 3)

                if is_table:
                    headers_row = [c.strip() for c in lines[0].split("|") if c.strip()]
                    rows = []
                    for tl in lines[2:]:
                        if "|" in tl:
                            row = [c.strip() for c in tl.split("|") if c.strip()]
                            rows.append(row)
                    if headers_row and rows:
                        pdf.add_table(headers_row, rows)
                else:
                    para_text = " ".join(l.strip() for l in lines)
                    pdf.add_body(para_text)

        # Insert figures after matching sections
        for section_match, figs in figure_map.items():
            if section_match in title_text:
                for fig_path, caption in figs:
                    pdf.add_figure(fig_path, caption)

        # Insert tables for specific sections
        if "Cross-species scaling" in title_text and "inversely proportional" in title_text:
            pdf.ln(2)

    # Save
    pdf.output(str(OUT_PDF))
    print(f"PDF saved to: {OUT_PDF}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    generate_all_figures()
    build_pdf()
    print("\nDone!")


if __name__ == "__main__":
    main()
