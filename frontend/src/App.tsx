import { useState, useCallback } from "react";
import type { ClassificationResult } from "./types/api";
import { classifyImage } from "./utils/api";
import UploadForm from "./components/UploadForm";
import ResultPanel from "./components/ResultPanel";
import "./App.css";

function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileSelect = useCallback((file: File | null) => {
    setSelectedFile(file);
    setResult(null);
    setError(null);
  }, []);

  const handleSubmit = useCallback(async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    try {
      const classificationResult = await classifyImage(selectedFile);
      setResult(classificationResult);
    } catch (err) {
      const message =
        err instanceof Error
          ? err.message
          : "An unexpected error occurred";

      setError(message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, [selectedFile]);

  return (
    <div className="app">
      <div className="background-glow" />

      <div className="container">
        <div className="header">
          <h1>AI Image Detector</h1>

          <p>
            Upload an image and detect whether it is
            AI-generated or authentic using deep learning
            and Grad-CAM explainability.
          </p>
        </div>

        <UploadForm
          selectedFile={selectedFile}
          loading={loading}
          error={error}
          onFileSelect={handleFileSelect}
          onSubmit={handleSubmit}
        />

        <ResultPanel
          result={result}
          error={error}
        />
      </div>
    </div>
  );
}

export default App;