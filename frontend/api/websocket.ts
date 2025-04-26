"use client"

// WebSocket client for real-time communication with the backend
import { useState, useEffect, useRef, useCallback } from "react"

export type WebSocketMessage = {
  type: string
  data: any
}

export type WebSocketStatus = "connecting" | "connected" | "disconnected" | "error"

export const useWebSocket = (notebookId: string) => {
  const [status, setStatus] = useState<WebSocketStatus>("disconnected")
  const [messages, setMessages] = useState<WebSocketMessage[]>([])
  const socketRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  const connect = useCallback(() => {
    if (socketRef.current?.readyState === WebSocket.OPEN) return

    // Close existing socket if any
    if (socketRef.current) {
      socketRef.current.close()
    }

    setStatus("connecting")

    // Connect to the WebSocket endpoint for the notebook
    const wsUrl = `${window.location.protocol === "https:" ? "wss:" : "ws:"}//${window.location.host}/ws/notebook/${notebookId}`
    const socket = new WebSocket(wsUrl)

    socket.onopen = () => {
      console.log("WebSocket connected")
      setStatus("connected")
    }

    socket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as WebSocketMessage
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

    socketRef.current = socket
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
      socketRef.current.send(JSON.stringify({ type, data }))
      return true
    }
    return false
  }, [])

  // Connect when the component mounts and disconnect when it unmounts
  useEffect(() => {
    connect()

    return () => {
      disconnect()
    }
  }, [connect, disconnect])

  return {
    status,
    messages,
    sendMessage,
    connect,
    disconnect,
  }
}
