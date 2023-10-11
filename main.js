import "./style.css"
import image from "/image-sharp.webp"
import { cleanupOpenCVObjects } from "./opencvUtils.js"
import * as cv from "@techstark/opencv-js"
const analysisParams = {
  sharpness: 100,
  motion: 0.5, // motion or defocus
  darknessPercentile: 90, // darkness
  darkness: 125,
  paramSpecular: 0.85, // specular
  thresholdForScaling: 99,
  paramScreen: 1.9, // screen
}

function defineKernels() {
  // Create a 3x3 matrix (kernel) for detecting horizontal edges (x-direction)
  const kx = cv.matFromArray(3, 3, cv.CV_64F, [0, 0, 0, 1, -2, 1, 0, 0, 0])

  // Create a 3x3 matrix (kernel) for detecting vertical edges (y-direction)
  const ky = cv.matFromArray(3, 3, cv.CV_64F, [0, 1, 0, 0, -2, 0, 0, 1, 0])
  // Create a 3x3 matrix (kernel) for detecting diagonal edges (upward)
  const ku = cv.matFromArray(3, 3, cv.CV_64F, [0, 0, 1, 0, -2, 0, 1, 0, 0])
  // Create a 3x3 matrix (kernel) for detecting diagonal edges (downward)
  const kv = cv.matFromArray(3, 3, cv.CV_64F, [1, 0, 0, 0, -2, 0, 0, 0, 1])
  // Return an array containing the defined kernels
  return [
    { name: "X", kernel: kx },
    { name: "Y", kernel: ky },
    { name: "U", kernel: ku },
    { name: "V", kernel: kv },
  ]
}

function convertToBGR(mat) {
  // Create a new empty matrix to store the result
  let mat_bgr = new cv.Mat()
  // Check if the input matrix has 4 channels (BGRA format)
  if (mat.channels() === 4) {
    // If it has 4 channels, convert it to BGR format
    cv.cvtColor(mat, mat_bgr, cv.COLOR_BGRA2BGR)
  } else {
    // If it has a different number of channels, simply clone the input matrix
    mat_bgr = mat.clone()
  }
  // Return the resulting BGR matrix
  return mat_bgr
}

function processImage(mat_bgr, kernel) {
  // Create a new empty matrix to store the processed image
  let img_processed = new cv.Mat()
  // Apply convolution to the input BGR image matrix using the provided kernel
  cv.filter2D(mat_bgr, img_processed, cv.CV_64F, kernel)
  // Return the processed image matrix
  return img_processed
}

function getChannel(matrix, channel) {
  return matrix.map((row) => row.map((cell) => cell[channel]))
}

function calculateVariance(arr) {
  // Get the number of elements in the input array 'arr'
  // Check if the array is empty
  if (arr.length === 0) {
    console.error("Array is empty.")
    return NaN
  }

  // Calculate the mean (average) of the values in the array
  const mean = arr.reduce((sum, val) => sum + val, 0) / arr.length

  // Calculate the sum of squared differences from the mean
  const squaredDifferencesSum = arr.reduce(
    (sum, val) => sum + Math.pow(val - mean, 2),
    0
  )

  // Calculate and return the variance (squared standard deviation)
  return squaredDifferencesSum / arr.length
}

function computeChannelSharpness(reshaped) {
  // iterating through the 2D array to extract the desired column

  // Create an array called 'columns' to store individual color channel data
  // const columns = [0, 1, 2].map((idx) => this.getColumn(reshaped, idx));

  // Calculate variances for each color channel
  // const variances = columns.map((column) =>
  //   this.calculateVariance(column.flat())
  // );
  // Return the maximum variance among the color channels, which represents sharpness
  // return Math.max(...variances);

  const channels = [0, 1, 2].map((channel) => getChannel(reshaped, channel))
  const variances = channels.map((channel) => calculateVariance(channel.flat()))
  return Math.max(...variances)
}

function computeSharpness(imgElement) {
  // Read the input image and store it in a matrix (mat)
  const mat = cv.imread(imgElement)

  // Define a set of kernels for image processing
  const kernels = defineKernels()

  // Convert the input image matrix (mat) to BGR color format
  const mat_bgr = convertToBGR(mat)

  // Initialize an array to store sharpness values for each kernel
  let sharpnessValues = []

  // Iterate through each kernel for image processing
  for (const entry of kernels) {
    // Process the image using the current kernel and get the result
    let img_processed = processImage(mat_bgr, entry.kernel)

    // Reshape the processed image data into a 3D array
    let reshaped = reshapeImage(img_processed)

    // Calculate the sharpness value for the current image channel
    let sharpness = computeChannelSharpness(reshaped)

    if (entry.name === "X") {
      sharpness *= 2
    }

    // Store the sharpness value in the array
    sharpnessValues.push(sharpness)
    console.log("sharpnessValues", sharpnessValues)

    // Clean up temporary OpenCV objects to avoid memory leaks
    cleanupOpenCVObjects(img_processed, reshaped)
  }
  // Clean up the remaining OpenCV objects
  cleanupOpenCVObjects(mat, mat_bgr, ...kernels)
  // Map sharpness values to fixed-precision numbers (6 decimal places)
  return sharpnessValues.map((val) => val.toFixed(6))
}

function reshapeImage(img_processed) {
  try {
    // Create a new MatVector to hold individual channels of the processed image
    let channels = new cv.MatVector()
    // Split the processed image into its individual color channels and store them in the MatVector
    cv.split(img_processed, channels)
    // Return the reshaped data by calling another function
    return reshapeFlattenedData(
      channels,
      img_processed.rows,
      img_processed.cols
    )
  } catch (error) {
    // Handle errors here
    console.log("Error in reshapeImage:", error)
    // You can choose to return an error object or throw the error further if needed.
    throw error
  }
}

function reshapeFlattenedData(channels, rows, cols) {
  let reshapedData = new Array(rows)
  let channels0 = channels.get(0).data64F
  let channels1 = channels.get(1).data64F
  let channels2 = channels.get(2).data64F

  for (let i = 0; i < rows; i++) {
    let row = new Array(cols)
    for (let j = 0; j < cols; j++) {
      row[j] = [
        channels0[i * cols + j],
        channels1[i * cols + j],
        channels2[i * cols + j],
      ]
    }
    reshapedData[i] = row
  }
  return reshapedData
}

function analyzeSharpness(sharpness) {
  let is_sharp = sharpness.every((val) => val > analysisParams.sharpness)
  let is_motion = false
  let is_blur = false

  if (!is_sharp) {
    let maxSharpness = Math.max(...sharpness)
    let minSharpness = Math.min(...sharpness)

    if ((maxSharpness - minSharpness) / maxSharpness > analysisParams.motion) {
      is_motion = true
    } else {
      is_blur = true
    }
  }

  return { is_sharp, is_motion, is_blur }
}

async function sharpness(image) {
  if (!cv) {
    throw new Error("OpenCV.js not loaded.")
  }
  // Compute sharpness values for the resized image

  const sharpnessValues = computeSharpness(image)
  // Analyze the sharpness values and store the result
  const result = analyzeSharpness(sharpnessValues)
  // Create a result object with sharpness, motion, and blur properties
  const sharpnessResult = {
    isBlur: result.is_blur,
    isMotion: result.is_motion,
    isSharp: result.is_sharp,
  }
  // Return the result object indicating image quality
  return sharpnessResult
}

document.querySelector("#app").innerHTML = `
  <div>
      <img id="image" src="${image}" class="logo" alt="car" />
  </div>
`

const imageElement = document.querySelector("#image")

// Call sharpness function every 1 second
setInterval(async () => {
  try {
    const sharpnessResult = await sharpness(imageElement)
    console.log(sharpnessResult)
  } catch (error) {
    console.error(error)
  }
}, 1000)
