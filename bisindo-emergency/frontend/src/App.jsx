/**
 * App — main application layout.
 * 
 * Layout:
 * ┌─────────────────────────────────────────┐
 * │    CameraView + SkeletonOverlay         │
 * │    (full width top, ~60% height)        │
 * ├──────────────────┬──────────────────────┤
 * │ DetectionDisplay │ EmergencyContact     │
 * │ (bottom-left)    │ (bottom-right)       │
 * └──────────────────┴──────────────────────┘
 * 
 * Registers service worker for push notifications.
 */
import { useEffect } from 'react'
import useWebSocket from './hooks/useWebSocket'
import CameraView from './components/CameraView'
import DetectionDisplay from './components/DetectionDisplay'
import EmergencyContact from './components/EmergencyContact'

export default function App() {
  const { isConnected, prediction, sendFrame } = useWebSocket()

  // Register service worker for push notifications
  useEffect(() => {
    if ('serviceWorker' in navigator) {
      navigator.serviceWorker.register('/sw.js')
        .then(async (registration) => {
          console.log('Service Worker registered')

          // Request push notification permission
          if ('PushManager' in window) {
            const permission = await Notification.requestPermission()
            if (permission === 'granted') {
              try {
                // Subscribe to push (using VAPID public key from server would go here)
                // For now, we just register the service worker
                const subscription = await registration.pushManager.getSubscription()
                if (subscription) {
                  // Send subscription to backend
                  await fetch('http://localhost:8000/subscribe-push', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(subscription.toJSON()),
                  })
                }
              } catch (e) {
                console.warn('Push subscription failed:', e)
              }
            }
          }
        })
        .catch((err) => {
          console.warn('Service Worker registration failed:', err)
        })
    }
  }, [])

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🤟 BISINDO Emergency Detection</h1>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            {isConnected ? 'Backend Connected' : 'Disconnected'}
          </span>
          <div className={`status-dot ${isConnected ? 'connected' : 'disconnected'}`} />
        </div>
      </header>

      <CameraView sendFrame={sendFrame} prediction={prediction} />

      <div className="bottom-panels">
        <DetectionDisplay prediction={prediction} />
        <EmergencyContact prediction={prediction} />
      </div>
    </div>
  )
}
