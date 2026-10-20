import type { ImageResponse } from "@/lib/api";

import { Verdict } from "./Verdict";

export function ImageResult({ data, previewUrl }: { data: ImageResponse; previewUrl: string | null }) {
  const { width, height } = data.image_size;

  return (
    <div className="result-stack">
      <Verdict
        label={data.label}
        probability={data.fake_probability}
        subject="Face classifier · ViT-Base"
        detail={[
          `${data.faces_analyzed} face${data.faces_analyzed === 1 ? "" : "s"} analysed; the verdict follows the most suspicious one.`,
          data.faces_skipped > 0 &&
            `${data.faces_skipped} small or partly out-of-frame face${data.faces_skipped === 1 ? " was" : "s were"} skipped.`,
        ]
          .filter(Boolean)
          .join(" ")}
      />

      {previewUrl && (
        <figure className="face-figure">
          <div className="face-stage">
            {/* eslint-disable-next-line @next/next/no-img-element -- local object URL */}
            <img src={previewUrl} alt="Uploaded image with detected faces outlined" />
            {data.faces.map((face, i) => (
              <div
                key={i}
                className={`face-box ${face.fake_probability >= 0.5 ? "face-box-fake" : "face-box-real"}`}
                style={{
                  left: `${(face.box.x / width) * 100}%`,
                  top: `${(face.box.y / height) * 100}%`,
                  width: `${(face.box.width / width) * 100}%`,
                  height: `${(face.box.height / height) * 100}%`,
                }}
              >
                <span className="face-tag mono">
                  {i + 1} · {face.fake_probability.toFixed(2)}
                </span>
              </div>
            ))}
          </div>
          <figcaption>Boxes show each detected face and its probability of being synthetic.</figcaption>
        </figure>
      )}
    </div>
  );
}
