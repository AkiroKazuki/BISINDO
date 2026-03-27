/**
 * SkeletonOverlay — draws MediaPipe skeleton on a canvas.
 *
 * Color-coded: Pose = blue, Left hand = green, Right hand = red.
 * Exported as a static utility (not a React component).
 */

// Colors per body part
const POSE_COLOR = 'rgba(59, 130, 246, 0.8)'       // Blue
const LEFT_HAND_COLOR = 'rgba(34, 197, 94, 0.8)'    // Green
const RIGHT_HAND_COLOR = 'rgba(239, 68, 68, 0.8)'   // Red

const POINT_RADIUS = 3
const LINE_WIDTH = 2

// MediaPipe connection indices
const POSE_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10],
  [11, 12], [11, 23], [12, 24], [23, 24],
  [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
  [24, 26], [26, 28], [28, 30], [28, 32], [30, 32],
]

const HAND_CONNECTIONS = [
  [0, 1], [0, 5], [0, 9], [0, 13], [0, 17],
  [1, 2], [2, 3], [3, 4],
  [5, 6], [6, 7], [7, 8],
  [9, 10], [10, 11], [11, 12],
  [13, 14], [14, 15], [15, 16],
  [17, 18], [18, 19], [19, 20],
]

function drawLandmarks(ctx, landmarks, color, width, height) {
  if (!landmarks) return

  ctx.fillStyle = color
  for (const lm of landmarks) {
    const x = lm.x * width
    const y = lm.y * height
    ctx.beginPath()
    ctx.arc(x, y, POINT_RADIUS, 0, 2 * Math.PI)
    ctx.fill()
  }
}

function drawConnections(ctx, landmarks, connections, color, width, height) {
  if (!landmarks) return

  ctx.strokeStyle = color
  ctx.lineWidth = LINE_WIDTH

  for (const [i, j] of connections) {
    if (!landmarks[i] || !landmarks[j]) continue
    const x1 = landmarks[i].x * width
    const y1 = landmarks[i].y * height
    const x2 = landmarks[j].x * width
    const y2 = landmarks[j].y * height

    ctx.beginPath()
    ctx.moveTo(x1, y1)
    ctx.lineTo(x2, y2)
    ctx.stroke()
  }
}

const SkeletonOverlay = {
  /**
   * Draw the full skeleton overlay on a canvas.
   *
   * @param {HTMLCanvasElement} canvas - Target canvas element
   * @param {Object} results - MediaPipe Holistic results
   * @param {HTMLVideoElement} video - Source video element
   */
  draw(canvas, results, video) {
    if (!canvas || !results || !video) return

    const ctx = canvas.getContext('2d')
    
    // Match canvas size to video display size
    const rect = video.getBoundingClientRect()
    canvas.width = rect.width
    canvas.height = rect.height

    const width = canvas.width
    const height = canvas.height

    // Clear previous frame
    ctx.clearRect(0, 0, width, height)

    // Draw pose
    drawConnections(ctx, results.poseLandmarks, POSE_CONNECTIONS, POSE_COLOR, width, height)
    drawLandmarks(ctx, results.poseLandmarks, POSE_COLOR, width, height)

    // Draw left hand
    drawConnections(ctx, results.leftHandLandmarks, HAND_CONNECTIONS, LEFT_HAND_COLOR, width, height)
    drawLandmarks(ctx, results.leftHandLandmarks, LEFT_HAND_COLOR, width, height)

    // Draw right hand
    drawConnections(ctx, results.rightHandLandmarks, HAND_CONNECTIONS, RIGHT_HAND_COLOR, width, height)
    drawLandmarks(ctx, results.rightHandLandmarks, RIGHT_HAND_COLOR, width, height)
  }
}

export default SkeletonOverlay
