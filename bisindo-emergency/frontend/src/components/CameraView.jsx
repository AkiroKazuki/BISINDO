/**
 * CameraView -- webcam video with skeleton overlay.
 * 
 * Uses useMediaPipe to extract keypoints and useWebSocket to stream to backend.
 * Triggers Web Speech API TTS when gesture is confirmed.
 */
import { useRef, useCallback, useEffect, useState } from 'react'
import useMediaPipe from '../hooks/useMediaPipe'
import SkeletonOverlay from './SkeletonOverlay'

export default function CameraView({ sendFrame, prediction }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const lastConfirmedRef = useRef(null)
  
  const [distanceWarning, setDistanceWarning] = useState(null)
  const [isIdle, setIsIdle] = useState(false)

  // Handle keypoint results from MediaPipe
  const handleResults = useCallback((keypoints, rawResults) => {
    // Update local UI states from metadata
    setDistanceWarning(rawResults.distanceWarning || null)
    setIsIdle(rawResults.isIdle || false)

    // ST-GCN requires continuous sliding window data. 
    // Gated Transmission: We ONLY send frame data if hands are active.
    // If the user has put their hands down for >1s, we cut transmission.
    if (!rawResults.isIdle) {
      sendFrame(keypoints)
    }

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
    <div className="camera-section" style={{ position: 'relative' }}>
      <video
        ref={videoRef}
        id="camera-feed"
        autoPlay
        playsInline
        muted
        style={{ display: isReady ? 'block' : 'none' }}
      />
      <canvas ref={canvasRef} id="skeleton-overlay" />
      
      {/* UI Overlays */}
      {isReady && distanceWarning && !isIdle && (
        <div style={{
          position: 'absolute', top: 20, left: '50%', transform: 'translateX(-50%)',
          background: 'rgba(239, 68, 68, 0.9)', color: 'white',
          padding: '12px 24px', borderRadius: '12px', fontWeight: 'bold',
          boxShadow: '0 4px 12px rgba(0,0,0,0.5)', zIndex: 10,
          whiteSpace: 'nowrap', backdropFilter: 'blur(4px)'
        }}>
          ⚠️ {distanceWarning}
        </div>
      )}

      {isReady && isIdle && (
        <div style={{
          position: 'absolute', bottom: 20, right: 20,
          background: 'rgba(30, 41, 59, 0.8)', color: 'white',
          padding: '8px 16px', borderRadius: '8px', fontSize: '0.9rem',
          display: 'flex', alignItems: 'center', gap: '8px', backdropFilter: 'blur(4px)'
        }}>
          <div style={{width: 8, height: 8, borderRadius: '50%', background: 'var(--text-muted)'}} />
          Standby (Menunggu Tangan)
        </div>
      )}

      {!isReady && !error && (
        <div className="camera-placeholder">
          <span>Memuat kamera dan MediaPipe...</span>
        </div>
      )}
      
      {error && (
        <div className="camera-placeholder" style={{ color: 'var(--accent-red)' }}>
          <span>{error}</span>
        </div>
      )}
    </div>
  )
}
