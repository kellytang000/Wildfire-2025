import "@/styles/globals.css";
import clsx from "clsx";
import { Metadata, Viewport } from "next";

import "@ant-design/v5-patch-for-react-19";
import { Providers } from "../context/providers";

import { siteConfig } from "@/config/site";

export const metadata: Metadata = {
  title: {
    default: siteConfig.name,
    template: `%s - ${siteConfig.name}`,
  },
  description: siteConfig.description,
  icons: {
    icon: "/favicon.ico",
  },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "white" },
    { media: "(prefers-color-scheme: dark)", color: "black" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html suppressHydrationWarning lang="zh-cn">
      <head />
      <body className="bg-white antialiased">
        <Providers themeProps={{ attribute: "class", defaultTheme: "light" }}>{children}</Providers>
      </body>
    </html>
  );
}
