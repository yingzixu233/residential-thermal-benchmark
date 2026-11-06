"""Build a three-page individual report from completed experiment artifacts."""
import argparse
from pathlib import Path
from xml.sax.saxutils import escape
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results")
    parser.add_argument("--output", default="report/individual_report.pdf")
    parser.add_argument("--author", default="Yingzi Xu")
    parser.add_argument("--repo-url", default="https://github.com/yingzixu233/residential-thermal-benchmark")
    args = parser.parse_args()
    root = ROOT/args.results
    output = ROOT/args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(root/"metrics.csv")
    sh = pd.read_csv(root/"shap/importance.csv")
    def stat(model, mode="rollout_path", days=10, noise=0, metric="rmse_C"):
        return m.query("model == @model and mode == @mode and days == @days and noise_C == @noise")[metric].agg(["mean", "std"])
    def fmt(model, mode, metric="rmse_C"):
        a=stat(model,mode,metric=metric)
        return f"{a['mean']:.3f} ({a['std']:.3f})"
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle",fontName="Helvetica-Bold",fontSize=18,leading=21,textColor=colors.HexColor("#12323f"),spaceAfter=8))
    styles.add(ParagraphStyle(name="ReportSub",fontName="Helvetica",fontSize=9.5,leading=13,textColor=colors.HexColor("#43565f"),spaceAfter=8))
    styles.add(ParagraphStyle(name="ReportBody",fontName="Helvetica",fontSize=10,leading=13,spaceAfter=7))
    styles.add(ParagraphStyle(name="ReportHead",fontName="Helvetica-Bold",fontSize=11.3,leading=14,spaceBefore=8,spaceAfter=5,textColor=colors.HexColor("#12323f")))
    styles.add(ParagraphStyle(name="ReportSmall",fontName="Helvetica",fontSize=8.2,leading=10.5,spaceAfter=5))
    styles.add(ParagraphStyle(name="ReportRef",fontName="Helvetica",fontSize=8,leading=10,spaceAfter=4))
    story=[]
    def add(text,style="ReportBody"):
        story.append(Paragraph(text,styles[style]))
    def head(text): add(text,"ReportHead")
    def figure(name, height):
        story.append(Image(str(root/"figures"/name),width=(450 if name == "shap_global.png" else 512),height=height))
    add("Residential Thermal Prediction<br/>with Limited and Noisy Sensor Data","ReportTitle")
    add(f"{escape(args.author)} | CEE 292: Scientific Machine Learning | Individual report | 6 November 2026","ReportSub")
    if args.repo_url:
        url=escape(args.repo_url)
        add(f'Code, data and complete results: <link href="{url}" color="#126078">{url}</link>',"ReportSmall")
    else:
        add("Code, data and complete results: prepared repository residential-thermal-benchmark; GitHub publication pending authentication.","ReportSmall")
    head("1. Research question and contribution")
    add("Can a small neural model predict residential temperatures reliably with only a few days of measurements, and do tree ensembles or noise augmentation help? Vasisht et al. [1] benchmark building thermal models. I implement their direct MLP transition, Section 3.2, Eq. (6), and extend it with Random Forest (RF), XGBoost, a temperature-noise ablation, and Shapley explanations. The contribution is an openly reproducible residential benchmark under scarce and perturbed inputs.")
    add("This is a <b>method-level adaptation</b>, not a reproduction of the full paper: its commercial-emulator data are available on request. The project instead follows the original outline's public nine-zone residential case [2,3]. It does not implement the paper's latent-state, NODE or physically constrained architectures, or its multi-step training curriculum.")
    head("2. Data and leakage-free evaluation")
    add("The unchanged house_data.csv contains 385 hourly observations from 30 March to 15 April 2019, without missing values. The input x(t) has 20 variables: nine temperatures (C), nine heating energies (Wh per hourly interval), outdoor temperature (C), and solar irradiance (W/m2). The output is the nine temperatures at t+1. Row-t heating/weather are aligned to that transition. The archive checksum and CSV SHA-256 are recorded.")
    add("Training uses observations [0,240), validation [240,288), and test [288,385), with zero-based, stop-exclusive indices. Recent 3/5/10-day training windows yield 71/119/239 pairs; validation and test yield 47 and 96. No transition crosses a split boundary. All scalers use training pairs only. Hyperparameters and MLP checkpoints minimize clean one-step validation RMSE; no validation refit or test-based tuning is used.")
    add("For each model, three seeds (0,1,2) are evaluated with added Gaussian indoor-temperature noise of SD 0, 0.2 or 0.5 C. Targets remain the recorded measurements. Noise realizations are paired across models and budgets. One-step evaluation refreshes the observed state each hour; 24-hour forecasts initialize once and recursively feed back predictions over all 73 complete test origins. Only the initial state is perturbed during rollout. Recorded future heating/weather are supplied, making these conditional open-loop forecasts.")
    head("3. Models and training")
    add("Direct transition: T(t+1) = f_theta([T(t), Q(t), T_out(t), GHI(t)]).", "ReportSmall")
    add("The MLP is a PyTorch 20-32-32-9 network with Tanh hidden activations and a linear output, predicting absolute next-hour temperatures. Two 32-unit layers follow the outline; the paper searches multiple widths and depths. Full-batch Adam uses learning rate 0.003 or 0.01, weight decay 10^-4, at most 1500 epochs, validation every 10 epochs, and 200-epoch patience. The augmented MLP adds fresh 0.2 C training-input noise each epoch; targets are unchanged.")
    add("Baselines are persistence and standardized Ridge (alpha = 0.1, 1, 10, 100). RF [5] uses 200 multi-output trees with (depth, leaf size) of (6,2) or (unlimited,3). XGBoost [4] fits nine scalar outputs, each using all 20 inputs: 250 trees, depth 2 or 3, learning rate 0.03, lambda 5 and subsample 0.9. These are small project-specific searches, not equal or exhaustive optimization budgets.")
    story.append(PageBreak())
    head("4. Results: accuracy, data budget and input noise")
    add("Table 1 reports mean (sample SD) over three seeds for ten-day training and clean inputs. RMSE/MAE pool equally over zones. Path RMSE pools all 73 origins x 24 horizons x 9 zones; terminal RMSE uses hour 24 only. Seed SD is descriptive, not a confidence interval. The deterministic baselines have zero clean-input SD.")
    add("<b>Table 1.</b> Clean test errors in C, with ten days of training.","ReportSmall")
    rows=[["Model","1 h RMSE","24 h path MAE","24 h path RMSE","h=24 RMSE"]]
    for model in ["Persistence","Ridge","MLP","MLP_noise","RF","XGBoost"]:
        rows.append([model.replace("MLP_noise","MLP + noise"),fmt(model,"one_step"),fmt(model,"rollout_path","mae_C"),fmt(model,"rollout_path"),fmt(model,"rollout_h24")])
    t=Table(rows,colWidths=[90,99,108,113,102],hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#12323f")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTNAME",(0,1),(-1,-1),"Helvetica"),("FONTSIZE",(0,0),(-1,-1),8.1),("BOTTOMPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),7),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#eff4f5"),colors.white]),("LINEBELOW",(0,-1),(-1,-1),.4,colors.HexColor("#b8c7cd"))]))
    story.append(t); story.append(Spacer(1,8))
    figure("accuracy.png",165)
    add("<b>Figure 1.</b> Clean-input path RMSE by recent training budget (left), and error by recursive horizon after ten-day training (right). Error bars/shading show seed SD. Horizon 1 uses 73 origins, while Table 1's one-step metric uses 96 transitions.","ReportSmall")
    ridge=stat("Ridge")["mean"]; mlp=stat("MLP")["mean"]; pers=stat("Persistence")["mean"]
    add(f"<b>Model complexity.</b> Ridge achieves the lowest clean ten-day path RMSE ({ridge:.3f} C), versus {mlp:.3f} C for MLP and about 1.011 C for both tree ensembles. Relative to persistence ({pers:.3f} C), Ridge reduces this error by {100*(1-ridge/pers):.1f}%. XGBoost's one-step error is lower than RF's, yet their path errors are nearly identical. One-step rankings therefore do not determine recursive forecast quality. These results support keeping a regularized linear baseline; they do not establish that linear models are generally superior.")
    m3=stat("MLP",days=3)["mean"]; n3=stat("MLP_noise",days=3)["mean"]
    add(f"<b>Limited data.</b> At three days, augmentation reduces MLP path RMSE from {m3:.3f} to {n3:.3f} C ({100*(1-n3/m3):.1f}%). At five days, the standard MLP reaches 0.522 C, compared with 0.577 C after augmentation; at ten days, errors are 0.627 and 0.630 C. The benefit is regime-dependent. Budgets change both sample count and operating conditions, and MLP accuracy is not monotonic in history length.")
    add("<b>Sensor perturbations.</b> With ten-day training and 0.5 C input noise, one-step RMSE is 0.495 C for augmented MLP, 0.514 C for standard MLP, and 0.803 C for Ridge. Ridge is sensitive to perturbed observed states, although its 24-hour path RMSE remains 0.530 C. Initialization noise is not re-injected inside recursive forecasts, so one-step and rollout robustness answer different questions. No physical clipping or stability constraint is imposed.")
    story.append(PageBreak())
    head("5. Shapley interpretation of the tree models")
    add("Interventional TreeSHAP [6] explains all 96 clean test inputs for the ten-day, seed-zero RF and XGBoost models, using the same 100 evenly spaced training inputs as background. It decomposes each one-step temperature prediction into a background expectation and feature contributions, in C. Global importance averages absolute contributions across test samples and all nine outputs. Reconstruction is checked for all 864 predictions per model; maximum errors are 5.19 x 10^-5 C (RF) and 2.78 x 10^-5 C (XGBoost).")
    figure("shap_global.png",150)
    add("<b>Figure 2.</b> Eight largest mean absolute SHAP contributions for each model; input names match the published CSV. These explain one-step outputs, not recursive trajectories.","ReportSmall")
    add("RF's largest mean attribution is T02 heating (0.317 C); XGBoost's is current T04 temperature (0.261 C). Indoor temperatures contribute 54.2% and 61.9% of total mean absolute attribution for RF and XGBoost; heating contributes 42.0% and 32.7%. These are model attribution shares, not physical energy shares or explained variance. Correlated temperatures can substitute for each other, and thermostat feedback couples heating to observed temperatures. Thus, a large cross-zone attribution neither proves thermal adjacency nor identifies a causal effect. The interventional background may also create unrealistic correlated-feature combinations.")
    head("6. Limitations and conclusion")
    add("The study has one house, about sixteen days, a two-day validation window, and strongly overlapping test rollouts. Three seeds measure algorithm/noise variability rather than independent building or seasonal uncertainty. Future forcing is assumed known, test targets contain measurement noise, and small search grids may disadvantage a model family. SHAP rankings are for one fitted seed and depend on the background. Claims about generalization, causal physics, or control performance require additional buildings, blocked temporal evaluations, and uncertain-forcing tests.")
    add("Within this protocol, the main finding is that a simple Ridge model gives the strongest clean-data performance, while noise augmentation improves selected scarce-data and noisy-input MLP cases. RF and XGBoost add useful comparison points but do not outperform the neural or linear models on long rollouts. The reproducible code, unchanged public data, all 162 evaluation settings, per-zone/per-horizon results, fitted reference models, and checked SHAP outputs are provided with the report.")
    head("References")
    refs=[
      '[1] S. Vasisht et al. Physics vs structure: A systematic benchmark of learning strategies for multi-zone building thermal dynamics. Energy and Buildings 361, 117472 (2026). <link href="https://doi.org/10.1016/j.enbuild.2026.117472">doi:10.1016/j.enbuild.2026.117472</link>.',
      '[2] C. Vallianos. Automated Multi-Zone Model Generation, v1.0 (2023). <link href="https://doi.org/10.5281/zenodo.10156745">doi:10.5281/zenodo.10156745</link>.',
      '[3] C. Vallianos, A. Athienitis and B. Delcroix. Automatic generation of multi-zone RC models using smart thermostat data from homes. Energy and Buildings 277, 112571 (2022). <link href="https://doi.org/10.1016/j.enbuild.2022.112571">doi:10.1016/j.enbuild.2022.112571</link>.',
      '[4] T. Chen and C. Guestrin. XGBoost: A Scalable Tree Boosting System. KDD (2016). <link href="https://doi.org/10.1145/2939672.2939785">doi:10.1145/2939672.2939785</link>.',
      '[5] L. Breiman. Random Forests. Machine Learning 45, 5-32 (2001). <link href="https://doi.org/10.1023/A:1010933404324">doi:10.1023/A:1010933404324</link>.',
      '[6] S. Lundberg et al. From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence 2, 56-67 (2020). <link href="https://doi.org/10.1038/s42256-019-0138-9">doi:10.1038/s42256-019-0138-9</link>.'
    ]
    for ref in refs: add(ref,"ReportRef")
    add("<b>Reproducibility note:</b> Implementation and manuscript prepared with AI assistance; reported experiments were executed from the included scripts. ","ReportSmall")
    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor("#c3ced2")); canvas.line(50,39,562,39)
        canvas.setFont("Helvetica",8); canvas.setFillColor(colors.HexColor("#60727a"))
        canvas.drawString(50,27,"CEE 292 | Residential thermal dynamics")
        canvas.drawRightString(562,27,f"{doc.page}")
    doc=SimpleDocTemplate(str(output),pagesize=letter,rightMargin=50,leftMargin=50,
                          topMargin=38,bottomMargin=49,title="Residential Thermal Prediction with Limited and Noisy Sensor Data",
                          author=args.author)
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    from pypdf import PdfReader
    n=len(PdfReader(output).pages)
    print(f"Wrote {output.name}: {n} pages")
    if n!=3: raise RuntimeError(f"Report must be three pages, got {n}; adjust layout")


if __name__=="__main__":
    main()
