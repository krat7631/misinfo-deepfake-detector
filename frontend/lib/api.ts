export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export type Label = "fake" | "real";

export type TokenAttribution = { token: string; contribution: number };

export type TextResponse = {
  result: {
    label: Label;
    fake_probability: number;
    confidence: number;
    tokens_analyzed: number;
    truncated: boolean;
  };
  attribution?: {
    method: string;
    base_value: number;
    tokens: TokenAttribution[];
    top_fake_signals: TokenAttribution[];
    top_real_signals: TokenAttribution[];
  };
  explanation?: string;
};

export type FaceBox = { x: number; y: number; width: number; height: number; score: number };

export type ImageResponse = {
  label: Label;
  fake_probability: number;
  faces_analyzed: number;
  faces_skipped: number;
  faces: { box: FaceBox; fake_probability: number }[];
  image_size: { width: number; height: number };
};

export type VideoResponse = {
  label: Label;
  fake_probability: number;
  most_suspicious_segment: { start: number; end: number };
  mean_frame_probability: number;
  max_frame_probability: number;
  share_of_frames_flagged: number;
  frames_sampled: number;
  frames_with_face: number;
  duration_seconds: number | null;
  timeline: { t: number; fake_probability: number }[];
};

export type Metrics = {
  samples: number;
  accuracy: number;
  f1_fake: number;
  roc_auc: number;
};

export type ModelCardResponse = {
  models: Record<
    string,
    {
      base_model: string;
      results: Record<string, Record<string, Metrics> | Metrics | number>;
      generators_seen_in_training?: string[] | null;
    }
  >;
  llm_explanations_enabled: boolean;
};

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

function errorMessage(payload: unknown, status: number): string {
  if (payload && typeof payload === "object" && "detail" in payload) {
    const detail = (payload as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail.map((d) => (d && typeof d === "object" && "msg" in d ? String(d.msg) : String(d))).join(" · ");
    }
  }
  return `Request failed with HTTP ${status}.`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    throw new ApiError(`Could not reach the API at ${API_BASE_URL}. Is the backend running?`, 0);
  }
  const text = await response.text();
  let payload: unknown = text;
  try {
    payload = JSON.parse(text);
  } catch {
    // Non-JSON body; keep the raw text for the error message.
  }
  if (!response.ok) throw new ApiError(errorMessage(payload, response.status), response.status);
  return payload as T;
}

export function analyzeText(text: string): Promise<TextResponse> {
  const form = new FormData();
  form.append("text", text);
  return request("/analyze-text", { method: "POST", body: form });
}

export function analyzeMedia<T>(endpoint: "analyze-image" | "analyze-video", file: File): Promise<T> {
  const form = new FormData();
  form.append("file", file);
  return request(`/${endpoint}`, { method: "POST", body: form });
}

export function fetchModelCard(): Promise<ModelCardResponse> {
  return request("/model-card");
}

export async function checkHealth(): Promise<boolean> {
  try {
    await request("/health");
    return true;
  } catch {
    return false;
  }
}
