const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, PageBreak
} = require("docx");

const BASE = "/Users/teo/Desktop/research/pi_tissue_paper";
const FIG_DIR = path.join(BASE, "manuscript", "figures");
const MD_PATH = path.join(BASE, "manuscript", "pi_tissue_paper_v3.md");
const OUT_PATH = path.join(BASE, "manuscript", "pi_tissue_paper_v3.docx");

// ── Figure mapping: section title substring → figure filename ──
const FIGURE_MAP = {
  "Tissue identity accounts for approximately 75%": {
    file: "fig1_core.png",
    caption: "Figure 1. Tissue identity is the dominant organizational mode of the human transcriptome and is near-invariant with age. (A) Stacked bar chart showing pi_tissue, pi_donor, and pi_residual for four age decades (GTEx v8, 263 donors, 6 tissues, 18,000 genes). (B) Line plot across decades. (C) Permutation null. (D) variancePartition REML confirmation."
  },
  "Single-cell validation across two platforms": {
    file: "fig2_v3_crossplatform.png",
    caption: "Figure 2. Cross-platform single-cell validation. (A) Gene detection QC: Smart-seq2 shows 29% decline vs 10% for 10x Chromium. (B) Cross-balanced 10x analysis: all 4 cell types show negative Delta_pi (mean = -0.07). (C) Platform comparison."
  },
  "Chromatin remodeling machinery erodes": {
    file: "fig3_chromatin.png",
    caption: "Figure 3. Chromatin remodeling genes erode tissue specificity 2.5x faster than expression-matched controls (p = 0.009)."
  },
  "Caloric restriction rescues pi through noise reduction": {
    file: "fig4_v3_bootstrap_cr.png",
    caption: "Figure 4. CR rescues 87% [82-91%] of pi loss via noise reduction. (A) pi values with bootstrap CIs. (B) Bootstrap rescue distribution (n=1000). (C) V_residual decreases in 100% of bootstraps."
  },
  "Cross-species scaling: erosion rate inversely proportional": {
    file: "fig5_v3_scaling.png",
    caption: "Figure 5. Cross-species scaling with macaque. (A) Macaque trajectory. (B) 4-species scaling law: alpha = -1.02. (C) Mouse bulk trajectory (corrected)."
  },
  "Cancer nearly abolishes tissue identity": {
    file: "figS2_pertissue_tcga.png",
    caption: "Figure S5. Cancer abolishes tissue identity: pi_tumor = 0.016 vs pi_normal ~ 0.73."
  }
};

// ── Parse markdown into structured sections ──
function parseMarkdown(mdText) {
  const lines = mdText.split("\n");
  const sections = [];
  let currentLevel = 0, currentTitle = "", currentBody = [];

  for (const line of lines) {
    if (line.startsWith("### ")) {
      if (currentTitle) sections.push({ level: currentLevel, title: currentTitle, body: currentBody.join("\n").trim() });
      currentLevel = 3; currentTitle = line.slice(4).trim(); currentBody = [];
    } else if (line.startsWith("## ")) {
      if (currentTitle) sections.push({ level: currentLevel, title: currentTitle, body: currentBody.join("\n").trim() });
      currentLevel = 2; currentTitle = line.slice(3).trim(); currentBody = [];
    } else if (line.startsWith("# ")) {
      if (currentTitle) sections.push({ level: currentLevel, title: currentTitle, body: currentBody.join("\n").trim() });
      currentLevel = 1; currentTitle = line.slice(2).trim(); currentBody = [];
    } else if (line.trim() === "---") {
      // skip separators
    } else {
      currentBody.push(line);
    }
  }
  if (currentTitle) sections.push({ level: currentLevel, title: currentTitle, body: currentBody.join("\n").trim() });
  return sections;
}

// ── Parse inline bold markers ──
function parseInlineFormatting(text) {
  const runs = [];
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  for (const part of parts) {
    if (part.startsWith("**") && part.endsWith("**")) {
      runs.push(new TextRun({ text: part.slice(2, -2), bold: true, font: "Times New Roman", size: 24 }));
    } else if (part) {
      runs.push(new TextRun({ text: part, font: "Times New Roman", size: 24 }));
    }
  }
  return runs;
}

// ── Parse a markdown table block ──
function parseTable(block) {
  const lines = block.trim().split("\n").filter(l => l.includes("|"));
  if (lines.length < 2) return null;

  const parseRow = (line) => line.split("|").map(c => c.trim()).filter(c => c && !c.match(/^-+$/));
  const headers = parseRow(lines[0]);
  // Skip separator line (lines[1])
  const rows = lines.slice(2).map(parseRow).filter(r => r.length > 0);

  if (headers.length === 0) return null;

  const contentWidth = 9360; // US Letter with 1" margins
  const colW = Math.floor(contentWidth / headers.length);
  const border = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
  const borders = { top: border, bottom: border, left: border, right: border };

  const headerRow = new TableRow({
    children: headers.map(h => new TableCell({
      borders,
      width: { size: colW, type: WidthType.DXA },
      shading: { fill: "D5E8D5", type: ShadingType.CLEAR },
      margins: { top: 60, bottom: 60, left: 80, right: 80 },
      children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: h, bold: true, font: "Times New Roman", size: 18 })]
      })]
    }))
  });

  const dataRows = rows.map((row, ri) => new TableRow({
    children: row.map(cell => new TableCell({
      borders,
      width: { size: colW, type: WidthType.DXA },
      shading: ri % 2 === 1 ? { fill: "F5F5F5", type: ShadingType.CLEAR } : undefined,
      margins: { top: 40, bottom: 40, left: 80, right: 80 },
      children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: cell, font: "Times New Roman", size: 18 })]
      })]
    }))
  }));

  const columnWidths = headers.map(() => colW);

  return new Table({
    width: { size: contentWidth, type: WidthType.DXA },
    columnWidths,
    rows: [headerRow, ...dataRows]
  });
}

// ── Build figure paragraph ──
function buildFigure(figFile, caption) {
  const figPath = path.join(FIG_DIR, figFile);
  if (!fs.existsSync(figPath)) {
    return [new Paragraph({
      children: [new TextRun({ text: `[Figure not found: ${figFile}]`, italics: true, color: "CC0000", font: "Times New Roman", size: 20 })]
    })];
  }

  const imgData = fs.readFileSync(figPath);
  const ext = path.extname(figFile).slice(1);

  return [
    new Paragraph({ spacing: { before: 200 } }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new ImageRun({
        type: ext,
        data: imgData,
        transformation: { width: 580, height: 220 },
        altText: { title: figFile, description: caption, name: figFile }
      })]
    }),
    new Paragraph({
      alignment: AlignmentType.JUSTIFIED,
      spacing: { before: 100, after: 200 },
      children: [new TextRun({ text: caption, italics: true, font: "Times New Roman", size: 18, color: "444444" })]
    })
  ];
}

// ── Convert a section body into docx paragraphs ──
function bodyToParagraphs(body) {
  if (!body) return [];
  const result = [];

  // Split by double newlines into blocks
  const blocks = body.split(/\n\s*\n/);
  for (const block of blocks) {
    const trimmed = block.trim();
    if (!trimmed) continue;

    // Check if block is a table
    const lines = trimmed.split("\n");
    if (lines.length >= 2 && lines[0].includes("|") && lines[1].includes("---") && lines[0].split("|").length >= 3) {
      const table = parseTable(trimmed);
      if (table) {
        result.push(table);
        continue;
      }
    }

    // Regular paragraph - join lines
    const text = lines.map(l => l.trim()).join(" ");
    const runs = parseInlineFormatting(text);
    result.push(new Paragraph({
      alignment: AlignmentType.JUSTIFIED,
      spacing: { after: 120, line: 360 },
      children: runs
    }));
  }
  return result;
}

// ── Main ──
async function main() {
  const mdText = fs.readFileSync(MD_PATH, "utf-8");
  const sections = parseMarkdown(mdText);
  console.log(`Parsed ${sections.length} sections`);

  const children = [];

  // Title
  children.push(new Paragraph({ spacing: { before: 600 } }));
  children.push(new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [new TextRun({
      text: "Tissue identity as a transcriptomic near-invariant:",
      bold: true, font: "Times New Roman", size: 32
    })]
  }));
  children.push(new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [new TextRun({
      text: "compositional drift, noise accumulation, and cross-species scaling",
      bold: true, font: "Times New Roman", size: 32
    })]
  }));
  children.push(new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 300, after: 100 },
    children: [new TextRun({ text: "[Authors to be completed]", font: "Times New Roman", size: 24 })]
  }));
  children.push(new Paragraph({ children: [new PageBreak()] }));

  // Process sections
  for (const sec of sections) {
    if (sec.title === "Authors" || sec.title === "[To be completed]") continue;
    // Skip the top-level title (already rendered)
    if (sec.level === 1) continue;

    // Major sections get page breaks
    if (sec.level === 2 && ["Introduction", "Results", "Discussion", "Methods", "Figure Legends", "Supplementary Figure Legends", "Supplementary Tables"].includes(sec.title)) {
      children.push(new Paragraph({ children: [new PageBreak()] }));
    }

    // Section heading
    let headingLevel;
    if (sec.level === 2) headingLevel = HeadingLevel.HEADING_1;
    else if (sec.level === 3) headingLevel = HeadingLevel.HEADING_2;

    if (sec.title !== "Abstract") {
      children.push(new Paragraph({
        heading: headingLevel,
        spacing: { before: 240, after: 120 },
        children: [new TextRun({
          text: sec.title,
          bold: true,
          font: "Times New Roman",
          size: sec.level === 2 ? 28 : 24
        })]
      }));
    } else {
      children.push(new Paragraph({
        heading: HeadingLevel.HEADING_1,
        spacing: { before: 240, after: 120 },
        children: [new TextRun({ text: "Abstract", bold: true, font: "Times New Roman", size: 28 })]
      }));
    }

    // Body paragraphs
    const bodyParas = bodyToParagraphs(sec.body);
    children.push(...bodyParas);

    // Insert figures after matching sections
    for (const [sectionMatch, figInfo] of Object.entries(FIGURE_MAP)) {
      if (sec.title.includes(sectionMatch)) {
        children.push(...buildFigure(figInfo.file, figInfo.caption));
      }
    }
  }

  // Build document
  const doc = new Document({
    styles: {
      default: {
        document: {
          run: { font: "Times New Roman", size: 24 }
        }
      },
      paragraphStyles: [
        {
          id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 28, bold: true, font: "Times New Roman" },
          paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 0 }
        },
        {
          id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
          run: { size: 24, bold: true, font: "Times New Roman", italics: true },
          paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 }
        }
      ]
    },
    sections: [{
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
        }
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            alignment: AlignmentType.LEFT,
            children: [new TextRun({ text: "pi_tissue manuscript v3", italics: true, font: "Times New Roman", size: 16, color: "888888" })]
          })]
        })
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [
              new TextRun({ text: "Page ", font: "Times New Roman", size: 16, color: "888888" }),
              new TextRun({ children: [PageNumber.CURRENT], font: "Times New Roman", size: 16, color: "888888" })
            ]
          })]
        })
      },
      children
    }]
  });

  const buffer = await Packer.toBuffer(doc);
  fs.writeFileSync(OUT_PATH, buffer);
  console.log(`DOCX saved to: ${OUT_PATH} (${(buffer.length / 1024).toFixed(0)} KB)`);
}

main().catch(err => { console.error(err); process.exit(1); });
