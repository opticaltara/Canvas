"use client"

import { useEffect } from "react"
import { useConnectionStore } from "@/store/connectionStore"

export default function StoreInitializer() {
  const loadConnections = useConnectionStore((state) => state.loadConnections)

  useEffect(() => {
    console.log("StoreInitializer: Calling loadConnections on mount...")
    loadConnections().then(() => {
      console.log("StoreInitializer: loadConnections promise resolved.")
    }).catch(err => {
      console.error("StoreInitializer: loadConnections promise rejected:", err)
    })
    // We only need to run this once when the app initializes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // Empty dependency array ensures it runs only once on mount

  // This component doesn't render anything visual
  return null 
} 