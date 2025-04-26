"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"

export default function Header() {
  const pathname = usePathname()

  return (
    <header className="border-b border-border bg-card">
      <div className="container mx-auto py-4 px-4">
        <Link href="/" className="text-xl font-bold text-primary">
          Sherlog Canvas
        </Link>
      </div>
      <nav className="container mx-auto px-4">
        <div className="flex">
          <Link
            href="/"
            className={`px-6 py-2 border-b-2 transition-all duration-200 ${
              pathname === "/"
                ? "border-primary text-primary font-medium"
                : "border-transparent hover:text-primary hover:border-primary/30"
            }`}
          >
            Canvases
          </Link>
          <Link
            href="/connections"
            className={`px-6 py-2 border-b-2 transition-all duration-200 ${
              pathname === "/connections"
                ? "border-primary text-primary font-medium"
                : "border-transparent hover:text-primary hover:border-primary/30"
            }`}
          >
            Data Connections
          </Link>
        </div>
      </nav>
    </header>
  )
}
