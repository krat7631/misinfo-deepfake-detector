import type { NextConfig } from "next";

const extraDevOrigins =
  process.env.NEXT_DEV_EXTRA_ORIGINS?.split(",")
    .map((s) => s.trim())
    .filter(Boolean) ?? [];

const nextConfig: NextConfig = {
  ...(extraDevOrigins.length
    ? { allowedDevOrigins: extraDevOrigins }
    : {}),
};

export default nextConfig;
