/**
 * useWebSocket — hook for WebSocket connection to the backend.
 * 
 * Connects to ws://localhost:8000/ws, exposes sendFrame() and prediction state.
 * Auto-reconnects on disconnect.
 */
import { useEffect, useRef, useState, useCallback } from 'react'

// Determine WebSocket protocol (wss for https, ws for http)
const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
// If on localhost (even on port 5173 dev server), point to backend port 8000. Else, use the current host.
const wsHost = window.location.hostname === 'localhost' ? 'localhost:8000' : window.location.host
const WS_URL = `${wsProtocol}//${wsHost}/ws`

const RECONNECT_DELAY = 3000 // 3 seconds

export default function useWebSocket() {
  const wsRef = useRef(null)
  const reconnectTimerRef = useRef(null)
  const [isConnected, setIsConnected] = useState(false)
  const [prediction, setPrediction] = useState(null)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    try {
      const ws = new WebSocket(WS_URL)

      ws.onopen = () => {
        console.log('WebSocket connected')
        setIsConnected(true)
        if (reconnectTimerRef.current) {
          clearTimeout(reconnectTimerRef.current)
          reconnectTimerRef.current = null
        }
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          setPrediction(data)
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e)
        }
      }

      ws.onclose = () => {
        console.log('WebSocket disconnected')
        setIsConnected(false)
        // Auto-reconnect
        reconnectTimerRef.current = setTimeout(() => {
          console.log('Attempting reconnect...')
          connect()
        }, RECONNECT_DELAY)
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        ws.close()
      }

      wsRef.current = ws
    } catch (e) {
      console.error('Failed to create WebSocket:', e)
      reconnectTimerRef.current = setTimeout(connect, RECONNECT_DELAY)
    }
  }, [])

  useEffect(() => {
    connect()

    return () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [connect])

  const sendFrame = useCallback((keypoints) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ keypoints }))
    }
  }, [])

  return { isConnected, prediction, sendFrame }
}
