"use client";

import { FormEvent, useEffect, useState } from "react";

import { ImageResult } from "@/components/ImageResult";
import { ModelCard } from "@/components/ModelCard";
import { TextResult } from "@/components/TextResult";
import { VideoResult } from "@/components/VideoResult";
import {
  analyzeMedia,
  analyzeText,
  ApiError,
  checkHealth,
  type ImageResponse,
  type TextResponse,
  type VideoResponse,
} from "@/lib/api";

type Mode = "text" | "image" | "video";

type Outcome =
  | { kind: "idle" }
  | { kind: "error"; message: string }
  | { kind: "text"; data: TextResponse }
  | { kind: "image"; data: ImageResponse; previewUrl: string }
  | { kind: "video"; data: VideoResponse };

const MODES: { id: Mode; label: string; hint: string }[] = [
  { id: "text", label: "Text", hint: "A headline, post or article. The model reads the first 256 tokens." },
  { id: "image", label: "Image", hint: "A photo containing at least one face. JPG, PNG or WebP, up to 25 MB." },
  { id: "video", label: "Video", hint: "A clip with a visible face. Frames are sampled every 0.5 s, up to 48." },
];

const SAMPLE_TEXT =
  "BREAKING: Scientists confirm that drinking hot water every hour completely cures viral infections, but the government is hiding it from you. Share before it gets deleted!";

export default function Home() {
  const [mode, setMode] = useState<Mode>("text");
  const [text, setText] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [outcome, setOutcome] = useState<Outcome>({ kind: "idle" });
  const [loading, setLoading] = useState(false);
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    checkHealth().then(setOnline);
  }, []);

  useEffect(() => {
    return () => {
      if (outcome.kind === "image") URL.revokeObjectURL(outcome.previewUrl);
    };
  }, [outcome]);

  function switchMode(next: Mode) {
    setMode(next);
    setFile(null);
    setOutcome({ kind: "idle" });
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    try {
      if (mode === "text") {
        setOutcome({ kind: "text", data: await analyzeText(text) });
      } else if (file && mode === "image") {
        const data = await analyzeMedia<ImageResponse>("analyze-image", file);
        setOutcome({ kind: "image", data, previewUrl: URL.createObjectURL(file) });
      } else if (file && mode === "video") {
        setOutcome({ kind: "video", data: await analyzeMedia<VideoResponse>("analyze-video", file) });
      }
    } catch (error) {
      setOutcome({ kind: "error", message: error instanceof ApiError ? error.message : String(error) });
    } finally {
      setLoading(false);
    }
  }

  const current = MODES.find((m) => m.id === mode)!;
  const canSubmit = !loading && (mode === "text" ? text.trim().length > 0 : file !== null);

  return (
    <main>
      <header className="masthead">
        <p className="eyebrow mono">
          <span className={`status-dot ${online ? "status-on" : online === false ? "status-off" : ""}`} aria-hidden />
          {online === null ? "Checking API…" : online ? "API online" : "API offline"}
        </p>
        <h1>Misinformation &amp; Deepfake Detector</h1>
        <p className="lede">
          A fine-tuned DistilBERT classifies text, and a fine-tuned Vision Transformer scores faces in images and
          video frames. Every prediction comes with the evidence behind it, and every model is tested on data it never
          saw during training.
        </p>
      </header>

      <section className="workbench" aria-labelledby="analyze-heading">
        <h2 id="analyze-heading" className="visually-hidden">Analyze content</h2>
        <div className="tabs" role="tablist" aria-label="Content type">
          {MODES.map((m) => (
            <button
              key={m.id}
              role="tab"
              aria-selected={mode === m.id}
              className={mode === m.id ? "tab tab-active" : "tab"}
              onClick={() => switchMode(m.id)}
              type="button"
            >
              {m.label}
            </button>
          ))}
        </div>

        <form onSubmit={handleSubmit} className="input-panel">
          <p className="hint">{current.hint}</p>
          {mode === "text" ? (
            <>
              <label htmlFor="text-input" className="visually-hidden">Text to analyse</label>
              <textarea
                id="text-input"
                rows={6}
                placeholder="Paste a headline, social post or article…"
                value={text}
                onChange={(e) => setText(e.target.value)}
                maxLength={20000}
              />
              <button type="button" className="link-button" onClick={() => setText(SAMPLE_TEXT)}>
                Use a sample claim
              </button>
            </>
          ) : (
            <label className="dropzone">
              <input
                type="file"
                accept={mode === "image" ? ".jpg,.jpeg,.png,.webp" : ".mp4,.mov,.avi,.mkv,.webm"}
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
              <span>{file ? file.name : `Choose ${mode === "image" ? "an image" : "a video"}`}</span>
            </label>
          )}
          <button className="primary" type="submit" disabled={!canSubmit}>
            {loading ? "Analysing…" : `Analyse ${current.label.toLowerCase()}`}
          </button>
        </form>

        <div className="result-area" aria-live="polite">
          {outcome.kind === "idle" && <p className="muted">Results appear here.</p>}
          {outcome.kind === "error" && <p className="error-banner">{outcome.message}</p>}
          {outcome.kind === "text" && <TextResult data={outcome.data} />}
          {outcome.kind === "image" && <ImageResult data={outcome.data} previewUrl={outcome.previewUrl} />}
          {outcome.kind === "video" && <VideoResult data={outcome.data} />}
        </div>
      </section>

      <section className="section" aria-labelledby="eval-heading">
        <h2 id="eval-heading">How well it works</h2>
        <p className="section-lede">
          Every test set is held out from training. DeepFakeFace rows use people the model never saw; rows marked
          “unseen domain” or “unseen generator” also come from sources it never trained on. Expect real-world
          performance to look more like those rows than the others.
        </p>
        <ModelCard />
      </section>

      <section className="section" aria-labelledby="limits-heading">
        <h2 id="limits-heading">Limitations</h2>
        <ul className="limits">
          <li>The text model judges writing style and framing. It does not check facts against sources.</li>
          <li>
            The image model was trained on StyleGAN faces, Stable Diffusion inpainting and InsightFace face swaps. On a
            generator it never saw (Stable Diffusion text-to-image) it catches about 6 in 10 fakes; newer generators may
            be harder still.
          </li>
          <li>Heavily resized, stretched or re-compressed images raise the false-positive rate on real photos.</li>
          <li>Only faces are analysed. Edited backgrounds, objects or captions are not.</li>
          <li>This is a research demo. Do not use it as the sole basis for decisions about people or content.</li>
        </ul>
      </section>
    </main>
  );
}
