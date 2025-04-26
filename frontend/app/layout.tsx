import type React from "react"
import type { Metadata } from "next"
import ClientRootLayout from "./client"

export const metadata: Metadata = {
  title: "Sherlog Canvas",
  description: "Interactive runbooks for incident management",
    generator: 'v0.dev'
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return <ClientRootLayout>{children}</ClientRootLayout>
}


import './globals.css'