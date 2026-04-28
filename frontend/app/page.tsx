"use client";

import { FormEvent, useMemo, useState } from "react";

import { ResultDisplay } from "@/components/ResultDisplay";

const baseUrlDefault = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function parseResponse(response: Response) {
  const text = await response.text();
  try {
    return JSON.parse(text) as unknown;
  } catch {
    return { raw: text };
  }
}

type Outcome =
  | { kind: "idle" }
  | { kind: "message"; text: string }
  | { kind: "response"; status: number; data: unknown };

export default function Home() {
  const [baseUrl, setBaseUrl] = useState(baseUrlDefault);
  const [text, setText] = useState("");
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [outcome, setOutcome] = useState<Outcome>({ kind: "idle" });
  const [loading, setLoading] = useState(false);

  const healthUrl = useMemo(() => `${baseUrl}/health`, [baseUrl]);

  async function handleTextSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    try {
      const form = new FormData();
      form.append("text", text);
      const response = await fetch(`${baseUrl}/analyze-text`, { method: "POST", body: form });
      const payload = await parseResponse(response);
      setOutcome({ kind: "response", status: response.status, data: payload });
    } catch (error) {
      setOutcome({ kind: "message", text: `Request failed: ${String(error)}` });
    } finally {
      setLoading(false);
    }
  }

  async function handleFileSubmit(endpoint: "analyze-image" | "analyze-video", file: File | null) {
    if (!file) {
      setOutcome({ kind: "message", text: "Please choose a file first." });
      return;
    }
    setLoading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch(`${baseUrl}/${endpoint}`, { method: "POST", body: form });
      const payload = await parseResponse(response);
      setOutcome({ kind: "response", status: response.status, data: payload });
    } catch (error) {
      setOutcome({ kind: "message", text: `Request failed: ${String(error)}` });
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <h1>AI Misinformation and Deepfake Detector</h1>
      <p>Portfolio-ready demo UI for text, image, and video analysis.</p>

      <section className="card">
        <label htmlFor="base-url">Backend URL</label>
        <input
          id="base-url"
          type="text"
          value={baseUrl}
          onChange={(e) => setBaseUrl(e.target.value.trim())}
          placeholder="http://localhost:8000"
        />
        <small>
          Health check: <a href={healthUrl}>{healthUrl}</a>
        </small>
      </section>

      <section className="card">
        <h2>Analyze text</h2>
        <form onSubmit={handleTextSubmit}>
          <textarea
            rows={5}
            placeholder="Paste a claim or post here..."
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <button disabled={loading || !text.trim()} type="submit">
            {loading ? "Analyzing..." : "Analyze Text"}
          </button>
        </form>
      </section>

      <section className="card">
        <h2>Analyze image</h2>
        <input
          type="file"
          accept=".jpg,.jpeg,.png,.webp"
          onChange={(e) => setImageFile(e.target.files?.[0] ?? null)}
        />
        <button disabled={loading} onClick={() => handleFileSubmit("analyze-image", imageFile)}>
          {loading ? "Analyzing..." : "Analyze Image"}
        </button>
      </section>

      <section className="card">
        <h2>Analyze video</h2>
        <input
          type="file"
          accept=".mp4,.mov,.avi,.mkv,.webm"
          onChange={(e) => setVideoFile(e.target.files?.[0] ?? null)}
        />
        <button disabled={loading} onClick={() => handleFileSubmit("analyze-video", videoFile)}>
          {loading ? "Analyzing..." : "Analyze Video"}
        </button>
      </section>

      <section className="card result-card">
        <h2>Latest result</h2>
        {outcome.kind === "idle" && <p className="result-muted">Run an analysis to see a formatted summary here.</p>}
        {outcome.kind === "message" && <p className="result-banner result-banner-warn">{outcome.text}</p>}
        {outcome.kind === "response" && (
          <ResultDisplay status={outcome.status} data={outcome.data} />
        )}
      </section>
    </main>
  );
}
