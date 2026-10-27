import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL(
    process.env.NEXT_PUBLIC_APP_URL || "https://handpose.ongyiumark.com"
  ),
  title: "Hand Pose Studio — 3D Dataset Builder & In-Browser ML",
  description:
    "Interactive 3D hand pose dataset builder and in-browser machine learning classifier powered by Google MediaPipe and TensorFlow.js.",
  icons: {
    icon: "/icon.png",
    apple: "/apple-icon.png",
  },
  openGraph: {
    title: "Hand Pose Studio — 3D Dataset Builder & In-Browser ML",
    description:
      "Interactive 3D hand pose dataset builder and in-browser machine learning classifier powered by Google MediaPipe and TensorFlow.js. 100% private in-browser local storage.",
    type: "website",
    images: [
      {
        url: "/opengraph-image.png",
        width: 1200,
        height: 630,
        alt: "Hand Pose Studio — 3D Hand Pose Dataset Builder & ML Classifier",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Hand Pose Studio — 3D Dataset Builder & In-Browser ML",
    description:
      "Interactive 3D hand pose dataset builder and in-browser machine learning classifier powered by Google MediaPipe and TensorFlow.js.",
    images: ["/opengraph-image.png"],
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://ko-fi.com" />
        <link
          rel="preconnect"
          href="https://js.stripe.com"
          crossOrigin="anonymous"
        />
        <link rel="dns-prefetch" href="https://js.stripe.com" />
        <link rel="dns-prefetch" href="https://m.stripe.com" />
      </head>
      <body className="min-h-screen bg-[#0c131d] text-slate-100 selection:bg-terracotta-500/30 selection:text-gold-200">
        {children}
      </body>
    </html>
  );
}
