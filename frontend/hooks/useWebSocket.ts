"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import { WS_URL } from "@/config/api-config"

console.log("Connecting websocket to WS_URL", WS_URL);

export type WebSocketStatus = "connecting" | "connected" | "disconnected" | "error"

export interface WebSocketMessage {
  type: string
  [key: string]: any
}

export type OnMessageCallback = (message: WebSocketMessage) => void;

export function useWebSocket(notebookId: string, onMessage: OnMessageCallback) {
  const [status, setStatus] = useState<WebSocketStatus>("disconnected")
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

    console.log(`Attempting WebSocket connection to: ${wsUrl}`) // More detailed log

    /* 
    // Temporarily disable PartySocket check to force native WebSocket
    if (typeof PartySocket !== 'undefined' && PartySocket) {
      socketRef.current = new PartySocket({
        host: WS_URL.replace(/^(ws|wss):\/\//, ""),
        room: `notebook-${notebookId}`,
      })
    } else { 
      socketRef.current = new WebSocket(wsUrl)
    }
    */
    // Always use native WebSocket for now
    // console.log("Forcing native WebSocket connection to:", wsUrl); // Add log
    try {
      socketRef.current = new WebSocket(wsUrl)
    } catch (err) {
      console.error(`Failed to create WebSocket object for URL: ${wsUrl}`, err);
      setStatus("error");
      // Optionally attempt reconnect or other error handling here
      return; // Exit connect function if creation fails
    }

    // Set up event handlers
    const socket = socketRef.current

    socket.onopen = (event: Event) => {
      console.log("WebSocket onopen event: Connection established", event)
      setStatus("connected")
      if (reconnectTimeoutRef.current) { // Clear reconnect timer on successful open
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
    }

    socket.onmessage = (event: MessageEvent) => {
      try {
        const message = JSON.parse(event.data) as WebSocketMessage
        console.log("WebSocket onmessage event: Message received:", message)
        onMessage(message) // Call the provided callback
      } catch (error) {
        console.error("WebSocket onmessage event: Failed to parse message:", error, "Raw data:", event.data)
      }
    }

    socket.onerror = (event: Event) => {
      console.error("WebSocket onerror event: Error occurred", event)
      setStatus("error")
      // Note: onerror is often followed by onclose. Reconnect logic is in onclose.
    }

    socket.onclose = (event: CloseEvent) => {
      console.log(`WebSocket onclose event: Connection closed. Code: ${event.code}, Reason: '${event.reason}', WasClean: ${event.wasClean}`, event)
      setStatus("disconnected")
      socketRef.current = null; // Clear the ref

      // Attempt to reconnect after a delay, only if not closed cleanly or if desired
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }

      console.log("Attempting WebSocket reconnect in 3 seconds...")
      reconnectTimeoutRef.current = setTimeout(() => {
        connect() // Call connect again
      }, 3000)
    }
  }, [notebookId, onMessage])

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
    console.log(`[useWebSocket useEffect] Running effect. notebookId: ${notebookId}`); // Use template literal
    if (notebookId) {
      console.log("[useWebSocket useEffect] notebookId is valid, calling connect..."); // Use double quotes
      connect()
    } else {
      console.log("[useWebSocket useEffect] notebookId is invalid, NOT calling connect."); // Use double quotes
    }

    return () => {
      console.log(`[useWebSocket useEffect] Cleanup running. notebookId: ${notebookId}`); // Use template literal
      disconnect()
    }
  }, [connect, disconnect, notebookId])

  return {
    status,
    // messages, // Removed messages state
    sendMessage,
    connect,
    disconnect,
  }
}
