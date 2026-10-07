"""Generate a detailed PDF describing the v3 prediction fix and follow-up work."""

import os
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_DIR = r"C:\DTI_GNN"
OUTPUT_PATH = os.path.join(
    PROJECT_DIR,
    "evaluation",
    "v3_results",
    "DTI_GNN_V3_Architecture_Changes_and_Results.pdf",
)


def footer(canvas, document):
    canvas.saveState()
    width, _ = A4
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(18 * mm, 12 * mm, "DTI-GNN v3 architecture changes")
    canvas.drawRightString(
        width - 18 * mm,
        12 * mm,
        f"Page {document.page}",
    )
    canvas.restoreState()


def make_table(rows, widths, header=True, font_size=8):
    table = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#172033")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands.extend([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ])
        if len(rows) > 1:
            commands.append(
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                    colors.white,
                    colors.HexColor("#F1F5F9"),
                ])
            )
    table.setStyle(TableStyle(commands))
    return table


def build_story(styles):
    story = []

    def para(text, style="BodyText"):
        story.append(Paragraph(text, styles[style]))
        story.append(Spacer(1, 4))

    def heading(text, level=1):
        story.append(Paragraph(text, styles[f"Heading{level}"]))
        story.append(Spacer(1, 4))

    story.append(Spacer(1, 20 * mm))
    story.append(Paragraph("DTI-GNN v3", styles["Title"]))
    story.append(Paragraph(
        "Architecture correction, prediction diagnosis, and local evaluation tools",
        styles["Subtitle"],
    ))
    story.append(Spacer(1, 8 * mm))
    story.append(HRFlowable(
        width="100%",
        thickness=1.2,
        color=colors.HexColor("#38BDF8"),
    ))
    story.append(Spacer(1, 8 * mm))
    para(
        "<b>Scope:</b> This document records the investigation from the original "
        "exploding predictions through the corrected architecture, subsequent "
        "cold-start results, and the training-sample diagnostic script."
    )
    para(
        "<b>Important distinction:</b> Fixing numerical instability makes the "
        "model output use the expected pIC50 scale. It does not by itself prove "
        "that the model predicts unseen interactions accurately."
    )

    heading("1. Initial problem: predictions exploded", 1)
    para(
        "The initial diagnostic showed actual pIC50 labels around 5–8, while "
        "predictions were in the thousands to tens of thousands. This was not "
        "a pIC50-versus-nM label conversion: the inspected training labels "
        "were already on a plausible pIC50 scale (approximately 3.0–12.7)."
    )
    para(
        "Inspection of <font name='Courier'>dti_model_v3_epoch15.pt</font> "
        "revealed that the first target-projection BatchNorm layer had "
        "<b>running variance exactly zero in all channels</b>. The next "
        "projection BatchNorm layer also had extremely small running variance."
    )
    para(
        "Training batches were grouped by target. Within a batch, the target "
        "encoder expanded one protein embedding to many duplicate examples and "
        "then passed those duplicate vectors through BatchNorm. The resulting "
        "running statistics were degenerate for protein-level inference. At "
        "evaluation time, normalization divided unseen protein representations "
        "by a near-zero standard deviation, amplifying activations and causing "
        "the huge predictions."
    )
    story.append(make_table([
        ["Observed evidence", "Meaning"],
        ["Training label range: 3.0 to about 12.7", "Targets were already pIC50-scale."],
        ["Old target projection running variance: 0.0", "BatchNorm inference normalization was degenerate."],
        ["Predictions in the thousands and above", "Consistent with near-zero-variance amplification."],
    ], [52 * mm, 116 * mm]))

    heading("2. Architecture correction: stable v3 / architecture version 2", 1)
    para(
        "The common model definition was moved into "
        "<font name='Courier'>model_v3.py</font> so training, evaluation, and "
        "the training-sample diagnostic build the same network."
    )
    story.append(make_table([
        ["Component", "Updated behavior"],
        ["Drug encoder", "Three GCNConv layers (18 atom features → 128 hidden channels), global mean pooling."],
        ["Target encoder", "ESM-2 35M layer-12 residue representations, mean-pooled per protein."],
        ["Target projection", "480 → 256 → 128 with LayerNorm, ReLU, and dropout; no BatchNorm running statistics."],
        ["Repeated targets", "Project each unique target once, then map the projected vector back to examples."],
        ["Fusion/head", "Concatenate 128-dimensional drug and target vectors; MLP 256 → 128 → 64 → 1 predicts pIC50."],
    ], [39 * mm, 129 * mm]))
    para(
        "The target projection change fixes the identified normalization failure. "
        "The GNN and prediction head were not fundamentally redesigned; the evidence "
        "pointed to target BatchNorm as the root cause."
    )

    heading("3. Training and checkpoint handling", 1)
    para(
        "<font name='Courier'>train_v3.py</font> imports the shared model, retains "
        "target-grouped batches and range-weighted squared error, and checks for "
        "non-finite outputs during training. Architecture version 2 checkpoints "
        "are saved under a separate stable prefix."
    )
    story.append(make_table([
        ["Old checkpoint family", "New checkpoint family"],
        ["dti_model_v3_epoch1.pt … epoch15.pt", "dti_model_v3_stable_epoch1.pt … epoch15.pt"],
        ["Unstable BatchNorm architecture", "LayerNorm architecture, version 2"],
        ["Left in place; not overwritten", "Used by updated evaluator"],
    ], [84 * mm, 84 * mm]))
    para(
        "The old files were preserved. The version-2 run produced "
        "<font name='Courier'>dti_model_v3_stable_epoch15.pt</font>, with the "
        "architecture version recorded in checkpoint metadata."
    )

    heading("4. Evaluation path and guards", 1)
    para(
        "<font name='Courier'>evaluate_v3.py</font> now loads the cold-start "
        "datasets from <font name='Courier'>C:\\DTI_GNN\\evaluation</font> (the "
        "actual location), rather than the training-data folder. It checks "
        "checkpoint architecture compatibility and stops if predictions are "
        "non-finite or have absolute magnitude over 20, a sanity threshold for "
        "the expected pIC50 scale. The generated report records the evaluation "
        "folder as the split-data source."
    )

    heading("5. Corrected cold-start results", 1)
    para(
        "The supplied report for stable epoch 15 shows numerically plausible "
        "predictions and substantially better overall behavior than the original "
        "exploding output. Results differ by split:"
    )
    story.append(make_table([
        ["Split", "N", "RMSE", "MAE", "Pearson r", "Within ±1.0"],
        ["Cold-drug", "138,556", "1.4835", "1.2105", "0.5528", "47.05%"],
        ["Cold-pair", "31,839", "1.8355", "1.4740", "0.2357", "41.58%"],
        ["Cold-target", "123,438", "1.8930", "1.5363", "0.2223", "38.93%"],
    ], [30 * mm, 24 * mm, 23 * mm, 23 * mm, 27 * mm, 41 * mm]))
    para(
        "Cold-drug generalization is strongest. Cold-pair and especially "
        "cold-target correlations are much lower, so overall valid-scale output "
        "should not be interpreted as strong ranking or affinity accuracy on "
        "unseen targets."
    )

    heading("6. High-pIC50 performance and data imbalance", 1)
    para(
        "Errors rise sharply for high actual pIC50 bins; predictions regress "
        "toward the middle, particularly for cold-target and cold-pair splits."
    )
    story.append(make_table([
        ["Actual pIC50 bin", "Train examples", "Cold-drug MAE", "Cold-pair MAE", "Cold-target MAE"],
        ["7.5–9.0", "143,839", "1.895", "2.514", "2.517"],
        ["9.0–10.5", "23,997", "2.742", "3.670", "3.769"],
        ["10.5–12.0", "739", "3.630", "5.520", "5.442"],
        ["12.0–13.5", "27", "not present in report bin", "6.439 (n=5)", "7.033 (n=3)"],
    ], [32 * mm, 28 * mm, 32 * mm, 34 * mm, 42 * mm]))
    para(
        "The training script already uses inverse-square-root label-bin weights, "
        "and the highest populated bin receives a large weight. However, a large "
        "weight cannot substitute for diversity: only 739 training examples lie "
        "from 10.5–12 and only 27 exceed 12. The cold-start test counts above "
        "12 are also tiny (3–5), so those metrics are highly uncertain."
    )
    para(
        "Conclusion from the report: acquire more quality-controlled high-pIC50 "
        "examples across different drugs and targets, then assess changes on a "
        "separate target-held-out validation set. Do not tune on the cold-start "
        "test report. No new high-score reweighting or architectural change was "
        "made because the report alone does not establish that such a change "
        "would improve generalization."
    )

    heading("7. Training-sample diagnostic added", 1)
    para(
        "<font name='Courier'>evaluate_train_sample_v3.py</font> loads the "
        "version-2 stable checkpoint and evaluates a reproducible sample of "
        "256 training examples. Sampling is stratified across pIC50 bins using "
        "seed 42, so the rare high-score ranges are included for inspection."
    )
    para(
        "It prints the first 20 actual/predicted pairs and the 10 largest "
        "absolute errors. Each row includes the target UniProt accession. The "
        "saved graph dataset contains atom features and bonds but no drug name "
        "or SMILES, so drug identity is represented by a SHA-256 fingerprint "
        "of the stored graph, not a chemical name."
    )
    story.append(make_table([
        ["Output file", "Contents"],
        ["train_sample_v3_report.txt", "Human-readable summary, first 20 rows, and worst 10 errors."],
        ["train_sample_v3_predictions.csv", "One row per sampled graph with IDs, labels, predictions, and absolute errors."],
        ["train_sample_v3_predictions.npz", "Sample indices, predictions, targets, and errors as NumPy arrays."],
    ], [69 * mm, 99 * mm]))
    para(
        "These are training examples, so the diagnostic is <b>in-sample</b>: it "
        "measures fit on examples the model trained on, not unbiased performance "
        "on unseen drugs/targets. Its deliberately stratified sample also "
        "over-represents rare high bins, so aggregate MAE/RMSE from it should "
        "not be compared directly with the full cold-start metrics."
    )

    heading("8. Files and reproducibility", 1)
    story.append(make_table([
        ["File", "Purpose"],
        ["model_v3.py", "Shared v2 architecture definition."],
        ["train_v3.py", "Training loop and stable checkpoint creation."],
        ["evaluate_v3.py", "Cold-start evaluation and report/CSV/NPZ output."],
        ["evaluate_train_sample_v3.py", "Stratified in-sample diagnostic."],
        ["checkpoints/dti_model_v3_stable_epoch15.pt", "Trained architecture-version-2 weights and metadata."],
        ["data/target_sequences.pkl", "Protein sequences used by ESM target encoder."],
        ["data/target_to_idx.pkl", "Target index-to-UniProt mapping used by diagnostics/training."],
        ["data/train_data_v2.pt and split files", "Training and evaluation graph datasets."],
    ], [74 * mm, 94 * mm]))
    para(
        "Code and checkpoint must stay together, along with compatible PyTorch, "
        "PyTorch Geometric, and ESM dependencies. The weights file alone does "
        "not contain the Python model implementation or protein sequences."
    )

    heading("9. Validation performed and what remains", 1)
    para(
        "Automated tests passed for finite target-encoder outputs with repeated "
        "proteins, a GNN forward/backward pass, reproducible stratified sampling "
        "that includes rare high scores, and stable graph fingerprints. Python "
        "files were syntax-compiled."
    )
    para(
        "The full model was not retrained or cold-start evaluated as part of "
        "creating this document. The report metrics above are transcribed from "
        "the supplied local evaluation report. The later training-sample "
        "diagnostic code was tested for sampling and identifiers, but it has "
        "not been run here against the GPU checkpoint."
    )
    para(
        "<b>Recommended next step:</b> run the training-sample script if an "
        "in-sample fit check is desired; for a trustworthy accuracy estimate, "
        "create a target-held-out validation split and report performance by "
        "pIC50 bin and split type."
    )

    story.append(Spacer(1, 5 * mm))
    story.append(HRFlowable(
        width="100%",
        thickness=0.7,
        color=colors.HexColor("#CBD5E1"),
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "Generated from the local project code and evaluation report. "
        "All results are pIC50-scale unless otherwise indicated.",
        styles["SmallNote"],
    ))
    return story


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    base_styles = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle(
            "ReportTitle",
            parent=base_styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=30,
            leading=36,
            textColor=colors.HexColor("#17365D"),
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "Subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=13,
            leading=18,
            textColor=colors.HexColor("#475569"),
            alignment=TA_CENTER,
        ),
        "Heading1": ParagraphStyle(
            "ReportHeading1",
            parent=base_styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#17365D"),
            spaceBefore=10,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "Heading2": ParagraphStyle(
            "ReportHeading2",
            parent=base_styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#2563EB"),
            keepWithNext=True,
        ),
        "BodyText": ParagraphStyle(
            "ReportBody",
            parent=base_styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13.5,
            textColor=colors.HexColor("#172033"),
            spaceAfter=6,
        ),
        "SmallNote": ParagraphStyle(
            "SmallNote",
            parent=base_styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#64748B"),
        ),
    }
    document = SimpleDocTemplate(
        OUTPUT_PATH,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=20 * mm,
        title="DTI-GNN v3 Architecture Changes and Results",
        author="DTI-GNN project",
    )
    document.build(
        build_story(styles),
        onFirstPage=footer,
        onLaterPages=footer,
    )
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
