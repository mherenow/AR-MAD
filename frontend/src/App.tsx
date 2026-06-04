/**
 * App Root Component
 *
 * Orchestrates global application state and coordinates the UploadForm and
 * ResultPanel components. Handles file selection, classification submission,
 * and state transitions between idle/loading/success/error.
 *
 * Requirements: 1.1, 1.6, 4.1, 4.6, 4.7
 */

import { useState, useCallback } from "react";
import type { ClassificationResult } from "./types/api";
import { classifyImage } from "./utils/api";
import UploadForm from "./components/UploadForm";
import ResultPanel from "./components/ResultPanel";
import "./App.css";

/**
 * App manages the four pieces of global state required by the classifier UI:
 *   - selectedFile: the File chosen by the user (null when none selected)
 *   - result:       the ClassificationResult from the last successful request
 *   - loading:      true while a /classify request is in-flight
 *   - error:        error message string when the last request failed
 */
function App() {
  // Task 11.1 — state initialised to null / false
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  /**
   * Task 11.2 — handleFileSelect
   *
   * Called by UploadForm whenever the user picks a new file (or clears the
   * picker). Updates the selectedFile state and clears any stale result/error
   * so the UI always reflects the currently selected image.
   *
   * Requirement 4.7: When a new image is selected the Result_Panel SHALL clear
   * all displayed result fields.
   */
  const handleFileSelect = useCallback((file: File | null) => {
    setSelectedFile(file);
    setResult(null);
    setError(null);
  }, []);

  /**
   * Task 11.3 — handleSubmit
   *
   * Async handler invoked when the user clicks "Classify Image". Drives the
   * full request lifecycle:
   *   1. Set loading → true, clear any previous error
   *   2. Call classifyImage() from the API client
   *   3. On success: store ClassificationResult
   *   4. On failure: store error message string
   *   5. Always: set loading → false
   *
   * Requirements 1.6, 4.6
   */
  const handleSubmit = useCallback(async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    try {
      const classificationResult = await classifyImage(selectedFile);
      setResult(classificationResult);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "An unexpected error occurred.";
      setError(message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, [selectedFile]);

  /**
   * Task 11.4 — wire up UploadForm and ResultPanel
   *
   * UploadForm receives all state and handler props it needs; ResultPanel
   * receives the result and error to decide what to render.
   *
   * Requirements 1.1, 4.1
   */
  return (
    <main>
      <h1>AI Image Classifier</h1>

      <UploadForm
        selectedFile={selectedFile}
        loading={loading}
        error={error}
        onFileSelect={handleFileSelect}
        onSubmit={handleSubmit}
      />

      <ResultPanel result={result} error={error} />
    </main>
  );
}

export default App;
