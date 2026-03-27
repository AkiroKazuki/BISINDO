/**
 * CameraView — webcam video with skeleton overlay.
 * 
 * Uses useMediaPipe to extract keypoints and useWebSocket to stream to backend.
 * Triggers Web Speech API TTS when gesture is confirmed.
 */
import { useRef, useCallback, useEffect } from 'react'
import useMediaPipe from '../hooks/useMediaPipe'
import SkeletonOverlay from './SkeletonOverlay'

export default function CameraView({ sendFrame, prediction }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const lastConfirmedRef = useRef(null)

  // Handle keypoint results from MediaPipe
  const handleResults = useCallback((keypoints, rawResults) => {
    // Send keypoints to backend via WebSocket
    sendFrame(keypoints)

    // Draw skeleton overlay
    if (canvasRef.current && videoRef.current) {
      SkeletonOverlay.draw(canvasRef.current, rawResults, videoRef.current)
    }
  }, [sendFrame])

  const { isReady, error } = useMediaPipe(videoRef, handleResults)

  // TTS on confirmed detection
  useEffect(() => {
    if (prediction?.is_confirmed && prediction?.class) {
      // Avoid repeating TTS for same confirmation
      const now = Date.now()
      if (lastConfirmedRef.current && now - lastConfirmedRef.current < 10000) {
        return
      }
      lastConfirmedRef.current = now

      // Web Speech API TTS
      if ('speechSynthesis' in window) {
        const ttsMessage = `Isyarat ${prediction.class} terdeteksi. Mengirim notifikasi darurat.`
        const utterance = new SpeechSynthesisUtterance(ttsMessage)
        utterance.lang = 'id-ID'
        utterance.rate = 1.0
        speechSynthesis.speak(utterance)
      }
    }
  }, [prediction?.is_confirmed, prediction?.class])

  return (
    <div className="camera-section">
      <video
        ref={videoRef}
        id="camera-feed"
        autoPlay
        playsInline
        muted
        style={{ display: isReady ? 'block' : 'none' }}
      />
      <canvas ref={canvasRef} id="skeleton-overlay" />
      
      {!isReady && !error && (
        <div className="camera-placeholder">
          <span>⏳ Memuat kamera dan MediaPipe...</span>
        </div>
      )}
      
      {error && (
        <div className="camera-placeholder" style={{ color: 'var(--accent-red)' }}>
          <span>❌ {error}</span>
        </div>
      )}
    </div>
  )
}
