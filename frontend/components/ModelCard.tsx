"use client";

import { useEffect, useState } from "react";

import { fetchModelCard, type Metrics, type ModelCardResponse } from "@/lib/api";

type Row = { model: string; testSet: string; metrics: Metrics; unseen: string | null };

const TEST_SET_NAMES: Record<string, string> = {
  welfake_test: "WELFake held-out articles",
  liar2_test_ood: "LIAR2 political claims",
  faces140k_test: "140k Faces held-out (StyleGAN)",
  dff_inpainting: "DeepFakeFace · SD inpainting",
  dff_insight: "DeepFakeFace · InsightFace swap",
  dff_text2img: "DeepFakeFace · SD text-to-image",
};

// Which generator each image test set comes from, to flag ones the model never trained on.
const IMAGE_GENERATORS: Record<string, string> = {
  faces140k_test: "StyleGAN",
  dff_inpainting: "inpainting",
  dff_insight: "insight",
  dff_text2img: "text2img",
};

function isMetrics(value: unknown): value is Metrics {
  return typeof value === "object" && value !== null && "accuracy" in value && "roc_auc" in value;
}

function collectRows(card: ModelCardResponse): Row[] {
  const rows: Row[] = [];
  const text = card.models.text_model_report?.results;
  if (text) {
    for (const [model, label] of [["tfidf_logreg_baseline", "TF-IDF + LogReg"], ["distilbert", "DistilBERT"]] as const) {
      const sets = text[model];
      if (!sets || isMetrics(sets) || typeof sets === "number") continue;
      for (const [key, metrics] of Object.entries(sets)) {
        if (isMetrics(metrics)) {
          rows.push({ model: label, testSet: TEST_SET_NAMES[key] ?? key, metrics, unseen: key.includes("ood") ? "unseen domain" : null });
        }
      }
    }
  }
  const imageCard = card.models.image_model_report;
  if (imageCard) {
    const seen = imageCard.generators_seen_in_training ?? [];
    for (const [key, metrics] of Object.entries(imageCard.results)) {
      if (isMetrics(metrics) && key in TEST_SET_NAMES) {
        const unseen = seen.includes(IMAGE_GENERATORS[key]) ? null : "unseen generator";
        rows.push({ model: "ViT-Base", testSet: TEST_SET_NAMES[key], metrics, unseen });
      }
    }
  }
  return rows;
}

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

export function ModelCard() {
  const [rows, setRows] = useState<Row[] | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    fetchModelCard().then((card) => setRows(collectRows(card))).catch(() => setFailed(true));
  }, []);

  if (failed) return <p className="muted">Evaluation results are unavailable while the API is offline.</p>;
  if (!rows) return <p className="muted">Loading evaluation results…</p>;
  if (rows.length === 0) return <p className="muted">No evaluation reports found in reports/.</p>;

  return (
    <div className="table-wrap">
      <table className="metrics">
        <thead>
          <tr>
            <th scope="col">Model</th>
            <th scope="col">Test set</th>
            <th scope="col" className="num">Samples</th>
            <th scope="col" className="num">Accuracy</th>
            <th scope="col" className="num">F1 (fake)</th>
            <th scope="col" className="num">ROC-AUC</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={`${row.model}-${row.testSet}`}>
              <td>{row.model}</td>
              <td>
                {row.testSet}
                {row.unseen && <span className="tag">{row.unseen}</span>}
              </td>
              <td className="num mono">{row.metrics.samples.toLocaleString()}</td>
              <td className="num mono">{pct(row.metrics.accuracy)}</td>
              <td className="num mono">{pct(row.metrics.f1_fake)}</td>
              <td className="num mono">{row.metrics.roc_auc.toFixed(3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
