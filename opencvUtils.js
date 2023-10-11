export function setupCounter(element) {
  let counter = 0
  const setCounter = (count) => {
    counter = count
    element.innerHTML = `count is ${counter}`
  }
  element.addEventListener("click", () => setCounter(counter + 1))
  setCounter(0)
}

export function cleanupOpenCVObjects(...objects) {
  // Iterate through the provided objects
  for (const object of objects) {
    // Check if the object exists and has a 'delete' method (typical for OpenCV objects)
    if (object && typeof object.delete === "function") {
      // Call the 'delete' method to release memory associated with the object
      object.delete()
    }
  }
}
