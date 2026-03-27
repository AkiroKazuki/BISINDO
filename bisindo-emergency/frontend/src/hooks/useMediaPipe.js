/**
 * useMediaPipe — hook for MediaPipe Holistic keypoint extraction.
 * 
 * Loads MediaPipe Holistic on mount, sets up camera loop,
 * and extracts 75 keypoints (33 pose + 21 left hand + 21 right hand) per frame.
 */
import { useEffect, useRef, useCallback, useState } from 'react'
import { Holistic } from '@mediapipe/holistic'
import { Camera } from '@mediapipe/camera_utils'

const NUM_POSE = 33
const NUM_HAND = 21
const TOTAL = NUM_POSE + NUM_HAND * 2 // 75

export default function useMediaPipe(videoRef, onResults) {
  const cameraRef = useRef(null)
  const holisticRef = useRef(null)
  const [isReady, setIsReady] = useState(false)
  const [error, setError] = useState(null)

  const extractKeypoints = useCallback((results) => {
    const keypoints = new Array(TOTAL)

    // Pose landmarks (0-32)
    for (let i = 0; i < NUM_POSE; i++) {
      if (results.poseLandmarks && results.poseLandmarks[i]) {
        const lm = results.poseLandmarks[i]
        keypoints[i] = [lm.x, lm.y, lm.z]
      } else {
        keypoints[i] = [0, 0, 0]
      }
    }

    // Left hand landmarks (33-53)
    for (let i = 0; i < NUM_HAND; i++) {
      if (results.leftHandLandmarks && results.leftHandLandmarks[i]) {
        const lm = results.leftHandLandmarks[i]
        keypoints[NUM_POSE + i] = [lm.x, lm.y, lm.z]
      } else {
        keypoints[NUM_POSE + i] = [0, 0, 0]
      }
    }

    // Right hand landmarks (54-74)
    for (let i = 0; i < NUM_HAND; i++) {
      if (results.rightHandLandmarks && results.rightHandLandmarks[i]) {
        const lm = results.rightHandLandmarks[i]
        keypoints[NUM_POSE + NUM_HAND + i] = [lm.x, lm.y, lm.z]
      } else {
        keypoints[NUM_POSE + NUM_HAND + i] = [0, 0, 0]
      }
    }

    return keypoints
  }, [])

  useEffect(() => {
    if (!videoRef.current) return

    // Initialize MediaPipe Holistic
    const holistic = new Holistic({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
      }
    })

    holistic.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    })

    holistic.onResults((results) => {
      const keypoints = extractKeypoints(results)
      if (onResults) {
        onResults(keypoints, results)
      }
    })

    holisticRef.current = holistic

    // Start camera
    const camera = new Camera(videoRef.current, {
      onFrame: async () => {
        if (holisticRef.current && videoRef.current) {
          await holisticRef.current.send({ image: videoRef.current })
        }
      },
      width: 1280,
      height: 720,
    })

    camera.start()
      .then(() => {
        setIsReady(true)
        setError(null)
      })
      .catch((err) => {
        console.error('Camera error:', err)
        setError('Failed to access camera. Please allow camera permissions.')
      })

    cameraRef.current = camera

    return () => {
      if (cameraRef.current) {
        cameraRef.current.stop()
      }
      if (holisticRef.current) {
        holisticRef.current.close()
      }
    }
  }, [videoRef, onResults, extractKeypoints])

  return { isReady, error }
}
