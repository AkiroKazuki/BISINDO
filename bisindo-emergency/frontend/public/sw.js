/**
 * Service Worker for BISINDO Emergency Detection.
 * Handles push notifications from the backend.
 */

self.addEventListener('push', (event) => {
  let data = { title: 'BISINDO Emergency', body: 'Isyarat darurat terdeteksi!' }

  try {
    data = event.data.json()
  } catch (e) {
    console.warn('Failed to parse push data:', e)
  }

  const options = {
    body: data.body,
    icon: data.icon || '/icon.png',
    badge: '/icon.png',
    vibrate: [200, 100, 200, 100, 200],
    tag: 'bisindo-emergency',
    renotify: true,
    requireInteraction: true,
    actions: [
      { action: 'open', title: 'Buka Aplikasi' },
      { action: 'dismiss', title: 'Tutup' },
    ],
  }

  event.waitUntil(
    self.registration.showNotification(data.title, options)
  )
})

self.addEventListener('notificationclick', (event) => {
  event.notification.close()

  if (event.action === 'open' || !event.action) {
    event.waitUntil(
      clients.openWindow('/')
    )
  }
})
