"use client";

type UnknownRecord = Record<string, unknown>;

function isRecord(v: unknown): v is UnknownRecord {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function num(v: unknown): number | undefined {
  if (typeof v === "number" && !Number.isNaN(v)) return v;
  return undefined;
}

function str(v: unknown): string | undefined {
  return typeof v === "string" ? v : undefined;
}

function EmotionBars({ emotion }: { emotion: UnknownRecord }) {
  const entries = Object.entries(emotion)
    .filter(([, v]) => typeof v === "number")
    .sort((a, b) => (b[1] as number) - (a[1] as number));

  return (
    <div className="result-emotions">
      {entries.map(([name, value]) => (
        <div key={name} className="result-meter-row">
          <span className="result-meter-label">{name}</span>
          <div className="result-meter-track">
            <div
              className="result-meter-fill"
              style={{ width: `${Math.min(100, Number(value))}%` }}
            />
          </div>
          <span className="result-meter-value">{Number(value).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}

function GenderRow({ gender }: { gender: UnknownRecord }) {
  const entries = Object.entries(gender).filter(([, v]) => typeof v === "number");
  return (
    <div className="result-tags">
      {entries.map(([k, v]) => (
        <span key={k} className="result-tag">
          {k}: {Number(v).toFixed(1)}%
        </span>
      ))}
    </div>
  );
}

export function ResultDisplay({
  status,
  data,
}: {
  status: number;
  data: unknown;
}) {
  if (data === null || data === undefined) {
    return <p className="result-muted">No response yet.</p>;
  }

  if (typeof data === "string") {
    return <pre className="result-raw">{data}</pre>;
  }

  if (!isRecord(data)) {
    return <pre className="result-raw">{JSON.stringify(data, null, 2)}</pre>;
  }

  const statusChip =
    status >= 400 ? (
      <span className="result-http result-http-error">HTTP {status}</span>
    ) : status !== 200 ? (
      <span className="result-http result-http-warn">HTTP {status}</span>
    ) : null;

  /* FastAPI validation / HTTP errors */
  if ("detail" in data) {
    const d = data.detail;
    const message = Array.isArray(d)
      ? d
          .map((e) => {
            if (typeof e === "string") return e;
            if (isRecord(e) && str(e.msg)) return str(e.msg)!;
            return JSON.stringify(e);
          })
          .join(" · ")
      : typeof d === "string"
        ? d
        : JSON.stringify(d);
    return (
      <div className="result-stack">
        {statusChip}
        <div className="result-banner result-banner-error">
          <p>{message}</p>
        </div>
      </div>
    );
  }

  /* Text + optional explanation */
  if ("result" in data && isRecord(data.result)) {
    const r = data.result as UnknownRecord;
    const label = str(r.label) ?? "—";
    const score = num(r.score);
    const explanation = str(data.explanation);
    const scorePct =
      score === undefined ? undefined : score <= 1 ? score * 100 : score;
    return (
      <div className="result-stack">
        {statusChip}
        <div className="result-banner result-banner-ok">
          <div className="result-kpi">
            <span className="result-kpi-label">Prediction</span>
            <span className="result-kpi-value">{label}</span>
          </div>
          {scorePct !== undefined && (
            <div className="result-kpi">
              <span className="result-kpi-label">Confidence</span>
              <span className="result-kpi-value">{scorePct.toFixed(1)}%</span>
            </div>
          )}
        </div>
        {explanation && (
          <div className="result-panel">
            <h3 className="result-panel-title">Explanation</h3>
            <p className="result-prose">{explanation}</p>
          </div>
        )}
        <details className="result-details">
          <summary>Raw JSON</summary>
          <pre className="result-raw">{JSON.stringify(data, null, 2)}</pre>
        </details>
      </div>
    );
  }

  /* Video heuristic (before generic "analysis" shapes) */
  if ("total_frames" in data && "suspicious_frames" in data && "is_deepfake" in data) {
    const total = num(data.total_frames) ?? 0;
    const susp = num(data.suspicious_frames) ?? 0;
    const flag = data.is_deepfake === true;
    return (
      <div className="result-stack">
        {statusChip}
        <div className={`result-banner ${flag ? "result-banner-warn" : "result-banner-ok"}`}>
          <span className="result-kpi-value">{flag ? "Flagged as suspicious" : "Below heuristic threshold"}</span>
        </div>
        <div className="result-grid">
          <div className="result-kpi result-kpi-inline">
            <span className="result-kpi-label">Frames sampled</span>
            <span className="result-kpi-value">{total}</span>
          </div>
          <div className="result-kpi result-kpi-inline">
            <span className="result-kpi-label">Suspicious frames</span>
            <span className="result-kpi-value">{susp}</span>
          </div>
        </div>
        <p className="result-note">
          Video scoring here is a simple heuristic, not a certified deepfake detector.
        </p>
        <details className="result-details">
          <summary>Raw JSON</summary>
          <pre className="result-raw">{JSON.stringify(data, null, 2)}</pre>
        </details>
      </div>
    );
  }

  /* Image: face analysis array */
  if ("analysis" in data) {
    const raw = data.analysis;
    const faces = Array.isArray(raw) ? raw : raw ? [raw] : [];
    const real = data.real === true;

    if (faces.length === 0 && str(data.error)) {
      return (
        <div className="result-stack">
          {statusChip}
          <div className="result-banner result-banner-warn">
            <p>{str(data.error)}</p>
          </div>
        </div>
      );
    }

    return (
      <div className="result-stack">
        {statusChip}
        <div className={`result-banner ${real ? "result-banner-ok" : "result-banner-warn"}`}>
          <span className="result-badge">{real ? "Analysis complete" : "Check response"}</span>
        </div>
        {faces.map((face, i) => {
          if (!isRecord(face)) return null;
          const emotion = face.emotion;
          const gender = face.gender;
          return (
            <div key={i} className="result-panel">
              <h3 className="result-panel-title">
                {faces.length > 1 ? `Face ${i + 1}` : "Face analysis"}
              </h3>
              <div className="result-grid">
                {str(face.dominant_emotion) && (
                  <div className="result-kpi result-kpi-inline">
                    <span className="result-kpi-label">Dominant emotion</span>
                    <span className="result-kpi-value">{str(face.dominant_emotion)}</span>
                  </div>
                )}
                {num(face.age) !== undefined && (
                  <div className="result-kpi result-kpi-inline">
                    <span className="result-kpi-label">Estimated age</span>
                    <span className="result-kpi-value">{num(face.age)}</span>
                  </div>
                )}
                {str(face.dominant_gender) && (
                  <div className="result-kpi result-kpi-inline">
                    <span className="result-kpi-label">Gender (model)</span>
                    <span className="result-kpi-value">{str(face.dominant_gender)}</span>
                  </div>
                )}
                {num(face.face_confidence) !== undefined && (
                  <div className="result-kpi result-kpi-inline">
                    <span className="result-kpi-label">Face confidence</span>
                    <span className="result-kpi-value">
                      {(() => {
                        const c = num(face.face_confidence)!;
                        return `${(c <= 1 ? c * 100 : c).toFixed(0)}%`;
                      })()}
                    </span>
                  </div>
                )}
              </div>
              {isRecord(emotion) && <EmotionBars emotion={emotion} />}
              {isRecord(gender) && (
                <>
                  <h4 className="result-subtitle">Gender scores</h4>
                  <GenderRow gender={gender} />
                </>
              )}
            </div>
          );
        })}
        <details className="result-details">
          <summary>Raw JSON</summary>
          <pre className="result-raw">{JSON.stringify(data, null, 2)}</pre>
        </details>
      </div>
    );
  }

  return (
    <div className="result-stack">
      {statusChip}
      <details className="result-details" open>
        <summary>Response ({status})</summary>
        <pre className="result-raw">{JSON.stringify(data, null, 2)}</pre>
      </details>
    </div>
  );
}
