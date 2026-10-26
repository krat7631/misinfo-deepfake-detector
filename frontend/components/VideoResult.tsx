import type { VideoResponse } from "@/lib/api";

import { Verdict } from "./Verdict";

const W = 640;
const H = 160;
const PAD = { left: 36, right: 12, top: 12, bottom: 24 };

type TimelineProps = {
  points: VideoResponse["timeline"];
  duration: number;
  segment: VideoResponse["most_suspicious_segment"];
};

function Timeline({ points, duration, segment }: TimelineProps) {
  const x = (t: number) => PAD.left + (t / Math.max(duration, 0.001)) * (W - PAD.left - PAD.right);
  const y = (p: number) => PAD.top + (1 - p) * (H - PAD.top - PAD.bottom);
  const path = points.map((p, i) => `${i === 0 ? "M" : "L"}${x(p.t).toFixed(1)},${y(p.fake_probability).toFixed(1)}`).join(" ");
  const segmentWidth = Math.max(x(segment.end) - x(segment.start), 6);

  return (
    <svg className="timeline" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Fake probability for each sampled frame over time">
      <rect
        x={x(segment.start) - 3}
        y={PAD.top}
        width={segmentWidth}
        height={H - PAD.top - PAD.bottom}
        className="tl-segment"
      />
      {[0, 0.5, 1].map((tick) => (
        <g key={tick}>
          <line x1={PAD.left} x2={W - PAD.right} y1={y(tick)} y2={y(tick)} className={tick === 0.5 ? "tl-threshold" : "tl-grid"} />
          <text x={PAD.left - 8} y={y(tick) + 4} textAnchor="end" className="tl-axis">{tick}</text>
        </g>
      ))}
      <path d={path} className="tl-line" />
      {points.map((p, i) => (
        <circle key={i} cx={x(p.t)} cy={y(p.fake_probability)} r={3}
          className={p.fake_probability >= 0.5 ? "tl-dot-fake" : "tl-dot-real"}>
          <title>{`${p.t}s · P(fake) ${p.fake_probability.toFixed(3)}`}</title>
        </circle>
      ))}
      <text x={PAD.left} y={H - 6} className="tl-axis">0s</text>
      <text x={W - PAD.right} y={H - 6} textAnchor="end" className="tl-axis">{duration.toFixed(1)}s</text>
    </svg>
  );
}

export function VideoResult({ data }: { data: VideoResponse }) {
  const duration = data.duration_seconds ?? data.timeline.at(-1)?.t ?? 1;
  const { start, end } = data.most_suspicious_segment;
  const flagged = Math.round(data.share_of_frames_flagged * 100);

  return (
    <div className="result-stack">
      <Verdict
        label={data.label}
        probability={data.fake_probability}
        subject="Frame-level face classifier"
        detail={`Scored by the most suspicious stretch, ${start.toFixed(1)}–${end.toFixed(1)} s. ${flagged}% of the ${data.frames_with_face} frames with a face scored above 0.5 (whole-clip mean ${data.mean_frame_probability.toFixed(2)}).`}
      />
      <section className="panel">
        <header className="panel-head">
          <h3>Per-frame timeline</h3>
          <p className="legend">
            <span className="swatch swatch-segment" /> most suspicious stretch
          </p>
        </header>
        <Timeline points={data.timeline} duration={duration} segment={data.most_suspicious_segment} />
      </section>
    </div>
  );
}
