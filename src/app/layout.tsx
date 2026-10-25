import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Gesture Studio — 3D Hand Pose Dataset Builder & In-Browser ML",
  description:
    "Interactive 3D hand pose dataset builder and in-browser machine learning classifier powered by Google MediaPipe and TensorFlow.js.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className="min-h-screen bg-slate-950 text-slate-100 selection:bg-cyan-500/30 selection:text-cyan-200">
        {children}
      </body>
    </html>
  );
}
