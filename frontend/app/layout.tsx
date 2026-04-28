import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "AI Misinfo and Deepfake Detector",
  description: "Demo app for misinformation and deepfake analysis.",
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
