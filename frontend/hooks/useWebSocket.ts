"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import { WS_URL } from "@/config/api-config"

// Check if PartySocket is available
let PartySocket: any
try {
  // Dynamic import to avoid issues with SSR
  PartySocket = require("partysocket")
} catch (e) {
  console.log("PartySocket not available, using native WebSocket")
}

export type WebSocketStatus = "connecting" | "connected" | "disconnected" | "error"

export interface WebSocketMessage {
  type: string
  [key: string]: any
}

export function useWebSocket(notebookId: string) {
  const [status, setStatus] = useState<WebSocketStatus>("disconnected")
  const [messages, setMessages] = useState<WebSocketMessage[]>([])
  const socketRef = useRef<WebSocket | any>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  const connect = useCallback(() => {
    if (socketRef.current?.readyState === WebSocket.OPEN) return

    // Close existing socket if any
    if (socketRef.current) {
      socketRef.current.close()
    }

    setStatus("connecting")

    // Connect to the WebSocket endpoint for the notebook
    const wsUrl = `${WS_URL}/ws/notebook/${notebookId}`

    // Use PartySocket if available, otherwise use native WebSocket
    if (PartySocket) {
      socketRef.current = new PartySocket({
        host: WS_URL.replace(/^(ws|wss):\/\//, ""),
        room: `notebook-${notebookId}`,
      })
    } else {
      socketRef.current = new WebSocket(wsUrl)
    }

    // Set up event handlers
    const socket = socketRef.current

    socket.onopen = () => {
      console.log("WebSocket connected")
      setStatus("connected")
    }

    socket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as WebSocketMessage
        console.log("WebSocket message received:", message)
        setMessages((prev) => [...prev, message])
      } catch (error) {
        console.error("Failed to parse WebSocket message:", error)
      }
    }

    socket.onerror = (error) => {
      console.error("WebSocket error:", error)
      setStatus("error")
    }

    socket.onclose = () => {
      console.log("WebSocket disconnected")
      setStatus("disconnected")

      // Attempt to reconnect after a delay
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }

      reconnectTimeoutRef.current = setTimeout(() => {
        connect()
      }, 3000)
    }
  }, [notebookId])

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      socketRef.current.close()
      socketRef.current = null
    }

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
      reconnectTimeoutRef.current = null
    }

    setStatus("disconnected")
  }, [])

  const sendMessage = useCallback((type: string, data: any) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      const message = {
        type,
        ...data,
        timestamp: Date.now(),
      }
      socketRef.current.send(JSON.stringify(message))
      return true
    }
    return false
  }, [])

  // Connect when the component mounts and disconnect when it unmounts
  useEffect(() => {
    if (notebookId) {
      connect()
    }

    return () => {
      disconnect()
    }
  }, [connect, disconnect, notebookId])

  return {
    status,
    messages,
    sendMessage,
    connect,
    disconnect,
  }
}
