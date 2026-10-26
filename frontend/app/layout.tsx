import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Misinformation & Deepfake Detector",
  description:
    "Fine-tuned DistilBERT and Vision Transformer models for fake-news text and synthetic-face detection, with SHAP explanations and out-of-domain evaluation.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
