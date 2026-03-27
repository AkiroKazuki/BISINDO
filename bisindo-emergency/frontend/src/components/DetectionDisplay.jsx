/**
 * DetectionDisplay — shows current detection status, predicted class, and confidence.
 * 
 * Status states:
 * - Mendeteksi... (gray) — buffer not full
 * - Terdeteksi: [CLASS] (yellow) — prediction, not yet confirmed
 * - DARURAT — Mengirim Notifikasi (red) — confirmed
 * - Notifikasi Terkirim (green) — cooldown active
 */

export default function DetectionDisplay({ prediction }) {
  const getStatus = () => {
    if (!prediction) {
      return { label: 'Mendeteksi...', icon: '', className: 'detecting' }
    }

    if (prediction.in_cooldown) {
      return { label: 'Notifikasi Terkirim', icon: '', className: 'sent' }
    }

    if (prediction.is_confirmed) {
      return {
        label: `DARURAT — ${prediction.class}`,
        icon: '',
        className: 'emergency'
      }
    }

    if (prediction.class && prediction.class !== 'NO_MODEL') {
      return {
        label: `Terdeteksi: ${prediction.class}`,
        icon: '',
        className: 'detected'
      }
    }

    if (!prediction.buffer_full) {
      return { label: 'Mengumpulkan frame...', icon: '', className: 'detecting' }
    }

    return { label: 'Mendeteksi...', icon: '', className: 'detecting' }
  }

  const status = getStatus()
  const confidence = prediction?.confidence || 0
  const confidencePercent = Math.round(confidence * 100)

  const getConfidenceClass = () => {
    if (confidence >= 0.85) return 'high'
    if (confidence >= 0.5) return 'medium'
    return 'low'
  }

  return (
    <div className={`detection-panel ${status.className === 'emergency' ? 'emergency' : ''}`} id="detection-display">
      <h2>Status Deteksi</h2>

      <div className={`status-badge ${status.className}`}>
        <span className="status-icon">{status.icon}</span>
        <span>{status.label}</span>
      </div>

      {prediction?.class && prediction.class !== 'NO_MODEL' && (
        <>
          <div className="prediction-class" style={{
            color: status.className === 'emergency' ? 'var(--accent-red)' :
                   status.className === 'detected' ? 'var(--accent-yellow)' :
                   'var(--text-primary)'
          }}>
            {prediction.class}
          </div>

          <div className="confidence-container">
            <div className="confidence-label">
              <span>Confidence</span>
              <span>{confidencePercent}%</span>
            </div>
            <div className="confidence-bar">
              <div
                className={`confidence-fill ${getConfidenceClass()}`}
                style={{ width: `${confidencePercent}%` }}
              />
            </div>
          </div>
        </>
      )}

      {!prediction?.buffer_full && prediction && (
        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textAlign: 'center' }}>
          Mengumpulkan frame untuk inferensi...
        </div>
      )}
    </div>
  )
}
