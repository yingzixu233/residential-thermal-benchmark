"""Build three pages of analysis with all four figures, plus a reference page."""
import argparse
from pathlib import Path
from xml.sax.saxutils import escape
import pandas as pd
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
    Spacer, Table, TableStyle, Image, PageBreak, KeepTogether)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter

ROOT = Path(__file__).resolve().parents[1]
MARGIN = 36  # 0.5 inch on both sides; figures and text share the same edges.
WIDTH = letter[0] - 2*MARGIN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--output", default="report/individual_report.pdf")
    parser.add_argument("--author", default="Yingzi Xu")
    parser.add_argument("--repo-url", default="https://github.com/yingzixu233/residential-thermal-benchmark")
    args = parser.parse_args()
    root, output = ROOT/args.results, ROOT/args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(root/"metrics.csv")
    def stat(model, mode="rollout_path", days=10, noise=0, metric="rmse_C"):
        return m.query("model == @model and mode == @mode and days == @days and noise_C == @noise")[metric].agg(["mean", "std"])
    def fmt(model, mode, metric="rmse_C"):
        a = stat(model, mode, metric=metric)
        return f"{a['mean']:.3f} ({a['std']:.3f})"
    styles = getSampleStyleSheet()
    definitions = [
        ("ReportTitle", "Helvetica-Bold", 17, 20, 6),
        ("ReportSub", "Helvetica", 9, 11, 5),
        ("ReportBody", "Helvetica", 9.8, 12.1, 5),
        ("ReportHead", "Helvetica-Bold", 11.2, 14, 4),
        ("ReportSmall", "Helvetica", 8.4, 10.5, 5),
        ("ReportRef", "Helvetica", 10, 14, 12),
    ]
    for name, font, size, leading, after in definitions:
        styles.add(ParagraphStyle(name=name, fontName=font, fontSize=size,
                                  leading=leading, spaceAfter=after,
                                  spaceBefore=6 if name == "ReportHead" else 0,
                                  textColor=colors.HexColor("#12323f") if name in
                                      ("ReportTitle", "ReportHead") else colors.black))
    story = []
    def add(text, style="ReportBody"):
        story.append(Paragraph(text, styles[style]))
    def head(text):
        add(text, "ReportHead")
    def figure(name, caption):
        img = Image(str(root/"figures"/name))
        ratio = WIDTH/img.imageWidth
        img.drawWidth, img.drawHeight = WIDTH, img.imageHeight*ratio
        story.append(KeepTogether([img, Spacer(1, 3),
                                   Paragraph(caption, styles["ReportSmall"])]))
    def page():
        story.append(PageBreak())

    # Page 1: methods and the full numerical comparison.
    add("Residential Thermal Prediction<br/>with Limited and Noisy Sensor Data", "ReportTitle")
    add(f"{escape(args.author)} | CEE 292: Scientific Machine Learning | Individual report | 6 November 2026", "ReportSub")
    if args.repo_url:
        url = escape(args.repo_url)
        add(f'Code, data and results: <link href="{url}" color="#126078">{url}</link>', "ReportSmall")
    head("1. Research question and contribution")
    add("Can a small neural model predict residential temperatures with only a few days of data, and do tree ensembles or noise augmentation help? I implement the direct MLP transition in Vasisht et al. [1], Section 3.2, Eq. (6), and extend it with Random Forest (RF), XGBoost, a temperature-noise ablation and Shapley explanations. This method-level adaptation follows the outline's public residential case [2,3]. The original commercial-emulator data are available on request; its full benchmark, latent-state and physics-constrained architectures, and multi-step curriculum are outside this implementation.")
    head("2. Data and evaluation")
    add("The unchanged house_data.csv has 385 hourly observations from 30 March to 15 April 2019, with no missing values. Twenty inputs comprise nine zone temperatures (C), nine heating energies (Wh per hourly interval), outdoor temperature (C) and solar irradiance (W/m2). Targets are the nine next-hour temperatures; row-t heating/weather are aligned to t+1. The public archive checksum and CSV SHA-256 are recorded.")
    add("Training observations are [0,240), validation [240,288), and test [288,385), using zero-based, stop-exclusive indices. Recent 3/5/10-day windows yield 71/119/239 training pairs; validation and test have 47 and 96. Split-boundary transitions are excluded. Scalers fit training pairs only. Hyperparameters and checkpoints minimize clean one-step validation RMSE, without validation refitting or test-based tuning.")
    add("Seeds 0,1,2 and Gaussian temperature-input noise of SD 0/0.2/0.5 C give 162 model-budget-seed-noise settings. Targets remain recorded measurements; noise is paired across models and budgets. One-step prediction refreshes measured inputs each hour. Recursive 24-hour forecasts use 73 complete, overlapping test origins, observe temperature only at initialization, and then feed back predictions. Noise is added only at initialization during rollout. Recorded future heating/weather are supplied, so forecasts are conditional on known forcing.")
    head("3. Models and training")
    add("Direct transition: T(t+1) = f_theta([T(t), Q(t), T_out(t), GHI(t)]).", "ReportSmall")
    add("The PyTorch MLP is 20-32-32-9 with Tanh hidden layers, a linear output, and training-only input/target standardization. It predicts absolute temperature. Full-batch Adam uses learning rate 0.003 or 0.01, weight decay 10^-4, at most 1500 epochs, checks every 10 epochs and 200-epoch patience. MLP + noise adds fresh 0.2 C Gaussian noise to training temperature inputs each epoch. The 32-unit layers follow the outline; the original paper searches multiple architectures.")
    add("Baselines are persistence and standardized Ridge (alpha 0.1/1/10/100). RF [5] has 200 multi-output trees; (depth, leaf size) is (6,2) or (unlimited,3). XGBoost [4] fits nine scalar outputs using all inputs: 250 trees, depth 2 or 3, learning rate 0.03, lambda 5, subsample 0.9. These small searches and the one-step objective are project choices, not exhaustive or equal optimization budgets.")
    head("4. Results overview")
    add("<b>Table 1.</b> Ten-day training, clean inputs: mean (sample SD) over three seeds, in C. Path metrics pool 73 origins x 24 horizons x 9 zones; h=24 uses the terminal hour. One-step metrics use 96 transitions. Seed SD is not a confidence interval.", "ReportSmall")
    rows = [["Model", "1 h RMSE", "24 h path MAE", "24 h path RMSE", "h=24 RMSE"]]
    for model in ["Persistence", "Ridge", "MLP", "MLP_noise", "RF", "XGBoost"]:
        rows.append([model.replace("MLP_noise", "MLP + noise"), fmt(model, "one_step"),
                     fmt(model, "rollout_path", "mae_C"), fmt(model, "rollout_path"),
                     fmt(model, "rollout_h24")])
    table = Table(rows, colWidths=[94, 108, 116, 116, 106], rowHeights=[20]+[18]*6)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#12323f")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE", (0,0), (-1,-1), 8.5),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#eff4f5"), colors.white]),
        ("LINEBELOW", (0,-1), (-1,-1), .4, colors.HexColor("#b8c7cd")),
    ]))
    story.append(table)
    page()

    # Page 2: all aggregate accuracy and robustness panels at full width.
    head("4. Results: data budget, forecast horizon and sensor noise")
    figure("accuracy.png", "<b>Figure 1.</b> Clean-input path RMSE versus recent training budget (left), and RMSE at each recursive horizon after ten-day training (right). Bars/shading show sample SD across three seeds. Horizon 1 uses 73 origins, so it differs from Table 1's 96-point one-step metric.")
    ridge, mlp, pers = stat("Ridge")["mean"], stat("MLP")["mean"], stat("Persistence")["mean"]
    add(f"<b>Complexity and long forecasts.</b> Ridge has the lowest clean ten-day path RMSE ({ridge:.3f} C), versus {mlp:.3f} C for MLP and about 1.011 C for both tree ensembles. This is {100*(1-ridge/pers):.1f}% below persistence ({pers:.3f} C). XGBoost beats RF at one step, but their path errors are nearly identical. One-step rankings therefore do not determine recursive accuracy; the linear baseline remains essential for this dataset.")
    m3, n3 = stat("MLP", days=3)["mean"], stat("MLP_noise", days=3)["mean"]
    add(f"<b>Limited data.</b> At three days, augmentation reduces MLP path RMSE from {m3:.3f} to {n3:.3f} C ({100*(1-n3/m3):.1f}%). At five days, standard and augmented MLP reach 0.522 and 0.577 C; at ten days, 0.627 and 0.630 C. Benefits depend on the regime. Changing the recent-data window changes both sample count and operating conditions, and accuracy is not monotonic in history length.")
    story.append(Spacer(1, 6))
    figure("noise_robustness.png", "<b>Figure 2.</b> Ten-day training under added indoor-temperature noise. One-step forecasts refresh perturbed measured states hourly (left); recursive forecasts perturb only the initial state (right). All model families use paired perturbations. Error bars show sample SD over three seeds.")
    add("<b>Noise robustness.</b> At 0.5 C input noise, one-step RMSE is 0.495 C for augmented MLP, 0.514 C for standard MLP, and 0.803 C for Ridge. Augmentation sacrifices some clean-input accuracy while reducing sensitivity to measurement perturbations. Ridge's recursive path RMSE is still 0.530 C. Since noise is not re-injected inside rollouts, the two panels address different deployment conditions. No model uses physical clipping or a stability constraint.")
    page()

    # Page 3: representative trajectories, explanations and qualified conclusions.
    head("5. Qualitative forecast trajectories")
    figure("example_rollout.png", "<b>Figure 3.</b> First test origin (11 April 2019, 00:00), ten-day training, seed 0, clean inputs: 24-hour predictions for zones 01, 05 and 09. Black curves are measured temperatures. The fixed example illustrates trajectory shape; aggregate results use all 73 origins and all nine zones.")
    add("The trajectory view complements aggregate errors by showing tracking and drift over time. It is an illustrative fixed case, not evidence that a model wins at every origin or zone. The repository retains predictions for every evaluated setting.")
    head("6. Shapley interpretation")
    figure("shap_global.png", "<b>Figure 4.</b> Top eight mean absolute TreeSHAP contributions for RF and XGBoost. Values are averaged over 96 clean test inputs and nine outputs for the ten-day, seed-zero models, with a common background of 100 evenly spaced training inputs.")
    add("Interventional TreeSHAP [6] decomposes one-step absolute temperature predictions into a background expectation and feature contributions (C). All 864 predictions/model pass reconstruction within 10^-4 C. RF's largest attribution is T02 heating (0.317 C); XGBoost's is T04 temperature (0.261 C). Temperature features account for 54.2%/61.9% of total mean absolute attribution, and heating for 42.0%/32.7% (RF/XGBoost). These are attribution shares, not energy shares or explained variance.")
    add("Correlated temperatures and thermostat feedback prevent causal interpretation: cross-zone importance does not establish thermal adjacency. Interventional masking can create unrealistic feature combinations, and rankings depend on the fitted seed and background. The explanations describe one-step outputs, not full recursive trajectories.")
    head("7. Limitations and conclusion")
    add("One house, about sixteen days, a two-day validation window and overlapping rollouts limit generalization. Three seeds capture algorithm/noise variability, not independent building or seasonal uncertainty. Future forcing is assumed known, labels contain measurement noise, and small search grids may disadvantage a family. Within this protocol, Ridge gives the strongest clean-input forecasts; noise augmentation helps selected scarce-data and noisy-input MLP cases. RF and XGBoost provide useful comparisons but do not outperform linear/neural models on long rollouts. Broader claims require more buildings, blocked temporal evaluations and uncertain-forcing tests.")
    page()

    # Page 4 is reserved for references, as requested.
    add("References", "ReportTitle")
    add("Supporting references for the three-page main report", "ReportSub")
    refs=[
      '[1] S. Vasisht et al. Physics vs structure: A systematic benchmark of learning strategies for multi-zone building thermal dynamics. Energy and Buildings 361, 117472 (2026). <link href="https://doi.org/10.1016/j.enbuild.2026.117472">doi:10.1016/j.enbuild.2026.117472</link>.',
      '[2] C. Vallianos. Automated Multi-Zone Model Generation, v1.0 (2023). <link href="https://doi.org/10.5281/zenodo.10156745">doi:10.5281/zenodo.10156745</link>.',
      '[3] C. Vallianos, A. Athienitis and B. Delcroix. Automatic generation of multi-zone RC models using smart thermostat data from homes. Energy and Buildings 277, 112571 (2022). <link href="https://doi.org/10.1016/j.enbuild.2022.112571">doi:10.1016/j.enbuild.2022.112571</link>.',
      '[4] T. Chen and C. Guestrin. XGBoost: A Scalable Tree Boosting System. KDD (2016). <link href="https://doi.org/10.1145/2939672.2939785">doi:10.1145/2939672.2939785</link>.',
      '[5] L. Breiman. Random Forests. Machine Learning 45, 5-32 (2001). <link href="https://doi.org/10.1023/A:1010933404324">doi:10.1023/A:1010933404324</link>.',
      '[6] S. Lundberg et al. From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence 2, 56-67 (2020). <link href="https://doi.org/10.1038/s42256-019-0138-9">doi:10.1038/s42256-019-0138-9</link>.'
    ]
    for ref in refs:
        add(ref, "ReportRef")
    story.append(Spacer(1, 10))
    add("<b>Reproducibility note.</b> Implementation and manuscript prepared with AI assistance. All reported experiments were executed from the included scripts. The public repository includes unchanged source data, all 162 evaluation settings, per-zone/per-horizon errors, fitted reference models and checked SHAP outputs.", "ReportSmall")
    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#c3ced2"))
        canvas.line(MARGIN, 29, letter[0]-MARGIN, 29)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#60727a"))
        canvas.drawString(MARGIN, 17, "CEE 292 | Residential thermal dynamics")
        canvas.drawRightString(letter[0]-MARGIN, 17, str(doc.page))
    doc = BaseDocTemplate(str(output), pagesize=letter,
                          title="Residential Thermal Prediction with Limited and Noisy Sensor Data",
                          author=args.author)
    frame = Frame(MARGIN, 38, WIDTH, letter[1]-30-38,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id="report", frames=[frame], onPage=footer))
    doc.build(story)
    from pypdf import PdfReader
    pdf = PdfReader(output)
    print(f"Wrote {output.name}: {len(pdf.pages)} pages")
    if len(pdf.pages) != 4:
        raise RuntimeError("Expected three content pages and one reference page; adjust layout")
    texts = [p.extract_text() for p in pdf.pages]
    expected = {0: ["Table 1."], 1: ["Figure 1.", "Figure 2."],
                2: ["Figure 3.", "Figure 4.", "7. Limitations"], 3: ["References", "[6]"]}
    for page_index, phrases in expected.items():
        for phrase in phrases:
            if phrase not in texts[page_index]:
                raise RuntimeError(f"Missing {phrase} on page {page_index+1}")
    print("Verified: four figures on content pages; references on page 4")


if __name__ == "__main__":
    main()
