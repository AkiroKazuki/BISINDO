import { useEffect, useRef, useCallback, useState } from 'react'
import { FilesetResolver, HolisticLandmarker } from '@mediapipe/tasks-vision'

const NUM_POSE = 33
const NUM_HAND = 21
const TOTAL = NUM_POSE + NUM_HAND * 2 // 75

// To prevent hands "teleporting" to the chest if MediaPipe loses them due to motion blur,
// we cache the last known positions.
let cachedLeftHand = new Array(NUM_HAND).fill([0, 0, 0])
let cachedRightHand = new Array(NUM_HAND).fill([0, 0, 0])

export default function useMediaPipe(videoRef, onResults) {
  const [isReady, setIsReady] = useState(false)
  const [error, setError] = useState(null)
  const landmarkerRef = useRef(null)
  const animationRef = useRef(null)
  const lastVideoTimeRef = useRef(-1)
  const framesWithoutHandsRef = useRef(0)

  const extractKeypoints = useCallback((result) => {
    const keypoints = new Array(TOTAL)
    let distanceWarning = null
    let hasHands = false

    // Pose landmarks (0-32)
    const pose = result.poseLandmarks?.[0]
    for (let i = 0; i < NUM_POSE; i++) {
      if (pose && pose[i]) {
        keypoints[i] = [pose[i].x, pose[i].y, pose[i].z]
      } else {
        keypoints[i] = [0, 0, 0]
      }
    }

    // Distance computation based on shoulders (indices 11 and 12)
    if (pose && pose[11] && pose[12]) {
      const dist = Math.sqrt(
        Math.pow(pose[11].x - pose[12].x, 2) + 
        Math.pow(pose[11].y - pose[12].y, 2)
      )
      // Thresholds: > 0.6 is too close, < 0.2 is too far
      if (dist > 0.6) {
        distanceWarning = "Posisi terlalu dekat, silakan mundur."
      } else if (dist < 0.2) {
        distanceWarning = "Posisi terlalu jauh, silakan mendekat."
      }
    }

    // Left hand landmarks (33-53)
    const leftHand = result.leftHandLandmarks?.[0]
    if (leftHand && leftHand.length > 0) {
      hasHands = true
      for (let i = 0; i < NUM_HAND; i++) {
        const p = leftHand[i]
        cachedLeftHand[i] = [p.x, p.y, p.z]
      }
    }
    for (let i = 0; i < NUM_HAND; i++) {
      keypoints[NUM_POSE + i] = cachedLeftHand[i]
    }

    // Right hand landmarks (54-74)
    const rightHand = result.rightHandLandmarks?.[0]
    if (rightHand && rightHand.length > 0) {
      hasHands = true
      for (let i = 0; i < NUM_HAND; i++) {
        const p = rightHand[i]
        cachedRightHand[i] = [p.x, p.y, p.z]
      }
    }
    for (let i = 0; i < NUM_HAND; i++) {
      keypoints[NUM_POSE + NUM_HAND + i] = cachedRightHand[i]
    }

    // Idle tracking
    if (!hasHands) {
      framesWithoutHandsRef.current += 1
    } else {
      framesWithoutHandsRef.current = 0
    }
    
    // If no hands detected for 30 consecutive frames (approx 1 second), consider idle
    const isIdle = framesWithoutHandsRef.current > 30

    return { keypoints, metadata: { isIdle, distanceWarning } }
  }, [])

  useEffect(() => {
    if (!videoRef.current) return

    let isRunning = true

    const initMediaPipe = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        )
        
        const landmarker = await HolisticLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "/holistic_landmarker.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minHandLandmarksConfidence: 0.1, // lowered to detect blurry fast movements
          minTrackingConfidence: 0.5,
          outputFaceBlendshapes: false,
          outputSegmentationMasks: false,
        })

        if (!isRunning) {
          landmarker.close()
          return
        }

        landmarkerRef.current = landmarker

        // Start WebCam with 3:4 aspect ratio to match training data (720x960)
        // This prevents the X/Y normalization from being stretched.
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: 480, height: 640, facingMode: "user" } 
        })
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          
          videoRef.current.onloadeddata = () => {
            setIsReady(true)
            
            // Log first detection to verify hand landmarks are detected
            let hasLoggedOnce = false
            
            const renderLoop = () => {
              if (!isRunning) return
              
              const video = videoRef.current
              if (video && video.readyState >= 2 && landmarkerRef.current) {
                const startTimeMs = performance.now()
                // Prevent duplicate processing of the same frame
                if (video.currentTime !== lastVideoTimeRef.current) {
                  const result = landmarkerRef.current.detectForVideo(video, startTimeMs)
                  
                  if (result) {
                    // Debug: log first result with hand landmarks
                    if (!hasLoggedOnce) {
                      const hasLeft = result.leftHandLandmarks && result.leftHandLandmarks.length > 0
                      const hasRight = result.rightHandLandmarks && result.rightHandLandmarks.length > 0
                      console.log('[MediaPipe] First detection:', {
                        pose: result.poseLandmarks?.length || 0,
                        leftHand: hasLeft ? result.leftHandLandmarks[0]?.length : 0,
                        rightHand: hasRight ? result.rightHandLandmarks[0]?.length : 0,
                      })
                      if (hasLeft || hasRight) {
                        hasLoggedOnce = true
                      }
                    }
                    
                    const { keypoints, metadata } = extractKeypoints(result)
                    if (onResults) {
                      // Pass raw results for skeleton overlay drawing, plus our rich metadata
                      onResults(keypoints, {
                        ...metadata,
                        poseLandmarks: result.poseLandmarks?.[0] || [],
                        leftHandLandmarks: result.leftHandLandmarks?.[0] || [],
                        rightHandLandmarks: result.rightHandLandmarks?.[0] || []
                      })
                    }
                  }
                  lastVideoTimeRef.current = video.currentTime
                }
              }
              animationRef.current = requestAnimationFrame(renderLoop)
            }
            
            renderLoop()
          }
        }
      } catch (err) {
        console.error('MediaPipe Init Error:', err)
        setError('Gagal memuat AI atau kamera. Pastikan izin kamera aktif.')
      }
    }

    initMediaPipe()

    return () => {
      isRunning = false
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
      if (videoRef.current && videoRef.current.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop())
      }
      if (landmarkerRef.current) {
        landmarkerRef.current.close()
      }
    }
  }, [videoRef, onResults, extractKeypoints])

  return { isReady, error }
}
