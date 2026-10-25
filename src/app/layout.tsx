import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Gesture Studio — 3D Hand Pose Dataset Builder & In-Browser ML",
  description:
    "Interactive 3D hand pose dataset builder and in-browser machine learning classifier powered by Google MediaPipe and TensorFlow.js.",
  icons: {
    icon: "/icon.png",
    apple: "/apple-icon.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className="min-h-screen bg-[#0c131d] text-slate-100 selection:bg-terracotta-500/30 selection:text-gold-200">
        {children}
      </body>
    </html>
  );
}
