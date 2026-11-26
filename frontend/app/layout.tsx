import type { Metadata, Viewport } from 'next';
import type { ReactNode } from 'react';
import '@fontsource-variable/noto-sans-kr';
import 'maplibre-gl/dist/maplibre-gl.css';
import './globals.css';

const title = 'SolarCast — 옥상 태양광 분석';
const description = '지도에서 옥상 태양광 설치 가능 면적과 예상 발전량을 살펴보는 분석 서비스의 Task 1 시안입니다.';
export const metadata: Metadata = { title, description, openGraph: { title, description, locale: 'ko_KR' } };

export const viewport: Viewport = { themeColor: '#111813' };

export default function RootLayout({ children }: { children: ReactNode }) {
  return <html lang="ko"><body>{children}</body></html>;
}
