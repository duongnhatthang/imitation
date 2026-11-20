"""Compile the current plot view into a bookmarked results PDF.

Run this on the execution machine after plotting, or use the automatic
publication option in publish_plot_view.py. Only rendered images are read.
"""

import argparse
import hashlib
import itertools
import os
import re
import tempfile
from pathlib import Path

OUTPUT_NAME = "imitation_learning_experiment_results.pdf"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
SEED_PATTERN = re.compile(r"_seed(\d+)")


def figure_pages(root):
    """Select current figures, grouping tasks and keeping coverage legible."""
    pages = []
    for family in ("classical", "atari"):
        base = root / family
        learning = sorted((base / "learning_curves").rglob("*"))
        images = [p for p in learning if p.suffix.lower() in IMAGE_SUFFIXES]
        comparisons = [p for p in images if p.stem.startswith("annotation-cost-")]

        def comparison_order(path):
            width = re.fullmatch(r"annotation-cost-w(\d+)", path.stem)
            return (0, -int(width[1])) if width else (1, path.as_posix())

        for path in sorted(comparisons, key=comparison_order):
            pages.append(("Annotation efficiency", family, [path]))
        remaining = [p for p in images if p not in comparisons]
        # The all-widths figure already contains every individual width.
        remaining = [
            p
            for p in remaining
            if "__student-width-" not in p.stem
            or not p.with_name(
                p.name.split("__student-width-")[0] + "__all-widths.png"
            ).is_file()
        ]
        for parent, group in itertools.groupby(remaining, key=lambda p: p.parent):
            group = list(group)
            # Full and masked three-metric figures stay together.
            if "state-binning" in parent.parts:
                pages.append(("State binning", family, group))
            else:
                for path in group:
                    pages.append(("Learning curves", family, [path]))

        runtime = sorted(
            p
            for p in (base / "runtime").rglob("*")
            if p.suffix.lower() in IMAGE_SUFFIXES
        )
        for index in range(0, len(runtime), 2):
            pages.append(("Runtime", family, runtime[index : index + 2]))

        coverage = sorted(
            p
            for p in (base / "coverage").rglob("*")
            if p.suffix.lower() in IMAGE_SUFFIXES
        )
        seeds = {}
        for path in coverage:
            match = SEED_PATTERN.search(path.stem)
            if match:
                key = (path.parent, path.stem[: match.start()])
                seeds[key] = min(seeds.get(key, int(match[1])), int(match[1]))
        selected = []
        for path in coverage:
            match = SEED_PATTERN.search(path.stem)
            if match:
                key = (path.parent, path.stem[: match.start()])
                if int(match[1]) != seeds[key]:
                    continue
            selected.append(path)
        for parent, group in itertools.groupby(selected, key=lambda p: p.parent):
            group = list(group)
            for index in range(0, len(group), 2):
                pages.append(("Coverage", family, group[index : index + 2]))

        diagnostics = sorted(
            p
            for p in (base / "diagnostics").rglob("*")
            if p.suffix.lower() in IMAGE_SUFFIXES
        )
        for path in diagnostics:
            pages.append(("Diagnostics", family, [path]))
    return pages


def caption(section):
    """Explain the review selection without inferring numerical findings."""
    return {
        "Annotation efficiency": (
            "BC-growing and interactive learning at matched training annotation budgets. "
            "Any crossing ratio refers to the first logged seed-mean crossing shown in the source figure."
        ),
        "State binning": (
            "Full and binned observations are shown together. Return, disagreement, and "
            "cross-entropy use the definitions, seed aggregation, masks, and budgets printed in the figures."
        ),
        "Learning curves": (
            "Student capacity or environment learning curves. Fixed BC is a terminal-budget "
            "reference. The figure records the methods, observations, budgets, and seed aggregation."
        ),
        "Runtime": (
            "Observed runtime from the execution machine. Read each figure's units, "
            "method labels, and acquisition accounting."
        ),
        "Coverage": (
            "Representative coverage at the lowest available training seed in each task and arm. "
            "Arrival maps and overlap maps are separate figures on the existing feature embedding. "
            "Every training seed remains available in the coverage directory."
        ),
        "Diagnostics": "Training diagnostics from the same published run.",
    }[section]


def build_report(plots_dir, run_id):
    """Write a deterministic review PDF and preserve any older PDF bytes."""
    from PIL import Image, ImageChops
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import A3, landscape, letter
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    from reportlab.platypus import Paragraph
    from xml.sax.saxutils import escape

    root = Path(plots_dir)
    pages = figure_pages(root)
    if not pages:
        raise ValueError("No current plot images were found")
    output = root / OUTPUT_NAME
    descriptor, name = tempfile.mkstemp(
        prefix=".results-", suffix=".pdf", dir=str(root)
    )
    os.close(descriptor)
    temporary = Path(name)
    text_style = ParagraphStyle(
        "caption",
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=HexColor("#334155"),
    )
    sections = []
    for section, family, _ in pages:
        label = family.capitalize() + " / " + section
        if label not in sections:
            sections.append(label)
    try:
        pdf = canvas.Canvas(
            str(temporary), pagesize=landscape(letter), invariant=1, pageCompression=1
        )
        pdf.setTitle("Imitation learning experiment results | " + run_id)
        pdf.setAuthor("Imitation learning experiments")
        pdf.setSubject("Learning curves, runtime, and representative coverage")
        width, height = landscape(letter)
        margin = 28

        def paragraph(text, x, top, available_width, style=text_style):
            block = Paragraph(escape(text), style)
            _, block_height = block.wrap(available_width, height)
            block.drawOn(pdf, x, top - block_height)
            return block_height

        def frame(title, page_number, total):
            pdf.setFillColor(HexColor("#0f172a"))
            title_style = ParagraphStyle(
                "title",
                fontName="Helvetica-Bold",
                fontSize=14,
                leading=16,
            )
            paragraph(title, margin, height - 20, width - 2 * margin, title_style)
            pdf.setFont("Helvetica", 9)
            pdf.setFillColor(HexColor("#475569"))
            pdf.drawString(margin, height - 66, run_id)
            pdf.drawRightString(width - margin, 18, f"{page_number} / {total}")
            pdf.drawString(margin, 18, "Imitation learning experiment results")

        total = len(pages) + 1
        frame("Experiment results", 1, total)
        paragraph(
            "A review of the current published plots. Learning curves and annotation comparisons "
            "appear first, followed by runtime, representative coverage, and diagnostics. "
            "Each figure keeps its original legends and scientific labels. Automatic return views can differ across tasks and conditions; read the panel axes.",
            margin,
            height - 90,
            width - 2 * margin,
        )
        paragraph(
            "Coverage includes the lowest available seed per task and arm. Open the adjacent "
            "coverage directory for every seed. Archives and superseded plots are excluded.",
            margin,
            height - 145,
            width - 2 * margin,
        )
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(margin, height - 200, "Contents")
        first_pages = {}
        for number, (section, family, _) in enumerate(pages, start=2):
            first_pages.setdefault(family.capitalize() + " / " + section, number)
        for index, (label, number) in enumerate(first_pages.items()):
            y = height - 225 - 22 * index
            pdf.setFont("Helvetica", 10)
            pdf.drawString(margin + 6, y, label)
            pdf.drawRightString(width - margin, y, str(number))
            pdf.linkAbsolute(
                label,
                f"section-{sections.index(label)}",
                (margin, y - 3, width - margin, y + 12),
            )
        pdf.bookmarkPage("contents")
        pdf.addOutlineEntry("Contents", "contents", level=0)
        pdf.showPage()

        seen = set()
        for number, (section, family, paths) in enumerate(pages, start=2):
            width, height = landscape(A3 if section == "Coverage" else letter)
            pdf.setPageSize((width, height))
            label = family.capitalize() + " / " + section
            title = section
            if len(paths) == 1:
                title += " / " + paths[0].stem.replace("__", " / ").replace("_", " ")
            frame(title, number, total)
            if label not in seen:
                key = f"section-{sections.index(label)}"
                pdf.bookmarkPage(key)
                pdf.addOutlineEntry(label, key, level=0)
                seen.add(label)
            figure_key = f"figure-{number - 1}"
            pdf.bookmarkPage(figure_key)
            pdf.addOutlineEntry(
                f"Figure {number - 1}: " + paths[0].stem, figure_key, level=1
            )
            # Each source is preserved as one image, with whitespace trimmed.
            image_top, image_bottom = height - 80, 116
            gap = 12
            slot_height = (image_top - image_bottom - gap * (len(paths) - 1)) / len(
                paths
            )
            for index, path in enumerate(paths):
                with Image.open(path) as source:
                    img = source.convert("RGB")
                    difference = ImageChops.difference(
                        img, Image.new("RGB", img.size, "white")
                    )
                    box = difference.getbbox()
                    if box:
                        img = img.crop(box)
                    if section == "Coverage":
                        slot_width = (
                            width - 2 * margin - gap * (len(paths) - 1)
                        ) / len(paths)
                        slot_top = image_top
                        slot_bottom = image_bottom
                        slot_left = margin + index * (slot_width + gap)
                    else:
                        slot_width = width - 2 * margin
                        slot_top = image_top - index * (slot_height + gap)
                        slot_bottom = slot_top - slot_height
                        slot_left = margin
                    ratio = min(
                        slot_width / img.width, (slot_top - slot_bottom) / img.height
                    )
                    draw_width, draw_height = img.width * ratio, img.height * ratio
                    x = slot_left + (slot_width - draw_width) / 2
                    y = (slot_bottom + slot_top - draw_height) / 2
                    pdf.drawImage(ImageReader(img), x, y, draw_width, draw_height)
            paragraph(
                f"Figure {number - 1}. " + caption(section),
                margin,
                100,
                width - 2 * margin,
            )
            source_text = "Sources: " + ", ".join(
                p.relative_to(root).as_posix() for p in paths
            )
            source_style = ParagraphStyle(
                "source", parent=text_style, fontSize=6.5, leading=8
            )
            paragraph(source_text, margin, 60, width - 2 * margin, source_style)
            pdf.showPage()
        pdf.save()
        if output.exists():
            previous = hashlib.sha256(output.read_bytes()).hexdigest()
            if previous == hashlib.sha256(temporary.read_bytes()).hexdigest():
                temporary.unlink()
                return output
            archive = root / "archive/previous-publications" / previous / OUTPUT_NAME
            archive.parent.mkdir(parents=True, exist_ok=True)
            if not archive.exists():
                archive.write_bytes(output.read_bytes())
        os.replace(str(temporary), str(output))
        return output
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    """Compile plots already rendered on the execution machine."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(build_report(args.plots_dir, args.run_id))


if __name__ == "__main__":
    main()
