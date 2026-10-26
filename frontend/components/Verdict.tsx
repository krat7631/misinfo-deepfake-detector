import type { Label } from "@/lib/api";

type Props = {
  label: Label;
  probability: number;
  subject: string;
  detail?: string;
};

function formatProbability(p: number): string {
  if (p > 0.999) return "> 0.999";
  if (p < 0.001) return "< 0.001";
  return `= ${p.toFixed(3)}`;
}

export function Verdict({ label, probability, subject, detail }: Props) {
  const percent = Math.round(probability * 100);
  return (
    <div className={`verdict verdict-${label}`}>
      <div className="verdict-head">
        <span className="verdict-kicker">{subject}</span>
        <span className="verdict-label">{label === "fake" ? "Likely fake" : "Likely real"}</span>
      </div>
      <div
        className="meter"
        role="meter"
        aria-label="Probability the content is fake"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent}
      >
        <div className="meter-fill" style={{ width: `${percent}%` }} />
        <div className="meter-threshold" aria-hidden />
      </div>
      <div className="verdict-scale">
        <span>Real</span>
        <span className="mono">P(fake) {formatProbability(probability)}</span>
        <span>Fake</span>
      </div>
      {detail && <p className="verdict-detail">{detail}</p>}
    </div>
  );
}
