import type React from "react"
import CanvasSidebar from "@/components/CanvasSidebar"

export default function CanvasRootLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-screen overflow-hidden">
      <CanvasSidebar />
      <div className="flex-1 overflow-auto">{children}</div>
    </div>
  )
} 