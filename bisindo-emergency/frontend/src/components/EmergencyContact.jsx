/**
 * EmergencyContact — form to manage emergency contacts.
 * 
 * Stores contacts in localStorage. When a gesture is confirmed,
 * contacts are used for SMS/voice notification via the backend.
 */
import { useState, useEffect, useCallback, useRef } from 'react'

const STORAGE_KEY = 'emergency_contacts'
const NOTIFICATION_COOLDOWN_MS = 12000 // slightly longer than backend's 10s cooldown

export default function EmergencyContact({ prediction, locationUrl }) {
  const [name, setName] = useState('')
  const [phone, setPhone] = useState('')
  const [contacts, setContacts] = useState([])
  const [notifyStatus, setNotifyStatus] = useState(null)
  const notificationSentRef = useRef(false)

  // Load contacts from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      if (stored) {
        setContacts(JSON.parse(stored))
      }
    } catch (e) {
      console.error('Failed to load contacts:', e)
    }
  }, [])

  // Save contacts to localStorage
  const saveContacts = useCallback((newContacts) => {
    setContacts(newContacts)
    localStorage.setItem(STORAGE_KEY, JSON.stringify(newContacts))
  }, [])

  // Add contact
  const handleAdd = (e) => {
    e.preventDefault()
    if (!name.trim() || !phone.trim()) return

    const newContact = {
      id: Date.now(),
      name: name.trim(),
      phone: phone.trim(),
    }

    saveContacts([...contacts, newContact])
    setName('')
    setPhone('')
  }

  // Delete contact
  const handleDelete = (id) => {
    saveContacts(contacts.filter(c => c.id !== id))
  }

  // Trigger notifications when gesture is confirmed (guarded by ref to prevent duplicates)
  useEffect(() => {
    if (prediction?.is_confirmed && prediction?.class
        && contacts.length > 0 && !notificationSentRef.current) {
      notificationSentRef.current = true
      triggerNotifications(prediction.class)
      setTimeout(() => { notificationSentRef.current = false }, NOTIFICATION_COOLDOWN_MS)
    }
  }, [prediction?.is_confirmed, prediction?.class])

  const triggerNotifications = async (gesture) => {
    setNotifyStatus('sending')

    for (const contact of contacts) {
      try {
        const httpHost = window.location.hostname === 'localhost' ? 'http://localhost:8000' : ''
        const notifyUrl = `${httpHost}/notify`
        
        const response = await fetch(notifyUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            gesture: gesture,
            contact_number: contact.phone,
            user_name: contact.name,
            location_url: locationUrl // Will be null if denied
          }),
        })

        if (response.ok) {
          setNotifyStatus('sent')
        } else {
          setNotifyStatus('error')
        }
      } catch (e) {
        console.error('Notification failed:', e)
        setNotifyStatus('error')
      }
    }

    // Reset status after 5 seconds
    setTimeout(() => setNotifyStatus(null), 5000)
  }

  return (
    <div className="contact-panel" id="emergency-contacts">
      <h2>Kontak Darurat</h2>

      <form className="contact-form" onSubmit={handleAdd}>
        <input
          id="contact-name-input"
          type="text"
          placeholder="Nama"
          value={name}
          onChange={(e) => setName(e.target.value)}
          autoComplete="name"
        />
        <input
          id="contact-phone-input"
          type="tel"
          placeholder="+628xxxxxxxxxx"
          value={phone}
          onChange={(e) => setPhone(e.target.value)}
          autoComplete="tel"
        />
        <button type="submit" className="btn-add" id="add-contact-btn">
          + Tambah Kontak
        </button>
      </form>

      <div className="contact-list">
        {contacts.map((contact) => (
          <div key={contact.id} className="contact-item">
            <div className="contact-info">
              <span className="contact-name">{contact.name}</span>
              <span className="contact-phone">{contact.phone}</span>
            </div>
            <button
              className="btn-delete"
              onClick={() => handleDelete(contact.id)}
              aria-label={`Hapus ${contact.name}`}
            >
              x
            </button>
          </div>
        ))}

        {contacts.length === 0 && (
          <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', textAlign: 'center', padding: '12px' }}>
            Belum ada kontak darurat. Tambahkan di atas.
          </div>
        )}
      </div>

      {notifyStatus && (
        <div style={{
          fontSize: '0.8rem',
          textAlign: 'center',
          padding: '8px',
          borderRadius: '8px',
          background: notifyStatus === 'sent' ? 'rgba(34, 197, 94, 0.15)' :
                      notifyStatus === 'error' ? 'rgba(239, 68, 68, 0.15)' :
                      'rgba(234, 179, 8, 0.15)',
          color: notifyStatus === 'sent' ? 'var(--accent-green)' :
                 notifyStatus === 'error' ? 'var(--accent-red)' :
                 'var(--accent-yellow)',
        }}>
          {notifyStatus === 'sending' && ' Mengirim notifikasi...'}
          {notifyStatus === 'sent' && ' Notifikasi terkirim!'}
          {notifyStatus === 'error' && ' Gagal mengirim notifikasi'}
        </div>
      )}
    </div>
  )
}
