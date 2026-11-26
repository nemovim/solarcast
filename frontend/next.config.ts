import type { NextConfig } from 'next';

const config: NextConfig = {
  allowedDevOrigins: ['*.e2b.app', '127.0.0.1'],
  devIndicators: false,
};

export default config;
