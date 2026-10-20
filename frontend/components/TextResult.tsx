import type { TextResponse, TokenAttribution } from "@/lib/api";

import { Verdict } from "./Verdict";

function tokenStyle(contribution: number, maxAbs: number): React.CSSProperties {
  const strength = maxAbs === 0 ? 0 : Math.min(1, Math.abs(contribution) / maxAbs);
  const alpha = (strength * 0.6).toFixed(3);
  return {
    backgroundColor: contribution > 0 ? `rgba(242, 166, 90, ${alpha})` : `rgba(147, 180, 255, ${alpha})`,
  };
}

function SignalList({ title, items }: { title: string; items: TokenAttribution[] }) {
  if (items.length === 0) return null;
  return (
    <div className="signal-list">
      <h4>{title}</h4>
      <ol>
        {items.slice(0, 6).map((item, i) => (
          <li key={`${item.token}-${i}`}>
            <span className="signal-token">{item.token.trim()}</span>
            <span className="mono">{item.contribution > 0 ? "+" : ""}{item.contribution.toFixed(3)}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

export function TextResult({ data }: { data: TextResponse }) {
  const { result, attribution, explanation } = data;
  const maxAbs = attribution
    ? Math.max(...attribution.tokens.map((t) => Math.abs(t.contribution)), 0)
    : 0;

  return (
    <div className="result-stack">
      <Verdict
        label={result.label}
        probability={result.fake_probability}
        subject="Text classifier · DistilBERT"
        detail={
          result.truncated
            ? `Only the first ${result.tokens_analyzed} tokens were analysed; the model reads at most 256.`
            : `${result.tokens_analyzed} tokens analysed.`
        }
      />

      {attribution && (
        <section className="panel">
          <header className="panel-head">
            <h3>Which words drove the prediction</h3>
            <p className="legend">
              <span className="swatch swatch-fake" /> towards fake
              <span className="swatch swatch-real" /> towards real
            </p>
          </header>
          <p className="attributed-text">
            {attribution.tokens.map((t, i) => (
              <span
                key={i}
                style={tokenStyle(t.contribution, maxAbs)}
                title={`${t.contribution > 0 ? "+" : ""}${t.contribution.toFixed(4)}`}
              >
                {t.token}
              </span>
            ))}
          </p>
          <div className="signal-grid">
            <SignalList title="Strongest fake signals" items={attribution.top_fake_signals} />
            <SignalList title="Strongest real signals" items={attribution.top_real_signals} />
          </div>
          <p className="footnote">
            {attribution.method}. Attributions explain what the model relied on, not whether a claim is true.
          </p>
        </section>
      )}

      {explanation && (
        <section className="panel">
          <h3>LLM summary (optional)</h3>
          <p>{explanation}</p>
        </section>
      )}
    </div>
  );
}
