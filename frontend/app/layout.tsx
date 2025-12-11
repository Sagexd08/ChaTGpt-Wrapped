import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ChatGPT Wrapped - Year in Chat Analytics",
  description: "Analyze your ChatGPT conversations and create a Spotify Wrapped-style recap",
  icons: {
    icon: '/favicon.ico',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-cyber-dark text-white antialiased">
        {children}
      </body>
    </html>
  );
}
