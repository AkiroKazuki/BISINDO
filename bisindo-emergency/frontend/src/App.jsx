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
import { useEffect, useState } from 'react'
import useWebSocket from './hooks/useWebSocket'
import CameraView from './components/CameraView'
import DetectionDisplay from './components/DetectionDisplay'
import EmergencyContact from './components/EmergencyContact'

export default function App() {
  const { isConnected, prediction, sendFrame } = useWebSocket()
  const [locationUrl, setLocationUrl] = useState(null)

  // Request Geolocation early
  useEffect(() => {
    if ('geolocation' in navigator) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const lat = position.coords.latitude
          const lng = position.coords.longitude
          setLocationUrl(`https://maps.google.com/?q=${lat},${lng}`)
        },
        (error) => console.warn("Geolocation denied or failed:", error),
        { enableHighAccuracy: true }
      )
    }
  }, [])

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
                // Fetch the VAPID public key from the backend
                const vapidResponse = await fetch('http://localhost:8000/vapid-public-key')
                const { publicKey } = await vapidResponse.json()

                if (!publicKey) {
                  console.warn('VAPID public key not configured on server')
                  return
                }

                // Subscribe to push notifications with the VAPID key
                const subscription = await registration.pushManager.subscribe({
                  userVisibleOnly: true,
                  applicationServerKey: publicKey,
                })

                // Send subscription to backend
                await fetch('http://localhost:8000/subscribe-push', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify(subscription.toJSON()),
                })
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
        <h1>BISINDO Emergency Detection</h1>
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
        <EmergencyContact prediction={prediction} locationUrl={locationUrl} />
      </div>
    </div>
  )
}
