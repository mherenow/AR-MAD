import type { ClassificationResult } from "../types/api";
import "./ResultPanel.css";

interface ResultPanelProps {
  result: ClassificationResult | null;
  error: string | null;
}

function formatPercentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export default function ResultPanel({
  result,
  error,
}: ResultPanelProps) {
  if (!result && !error) {
    return (
      <div className="glass-card result-panel empty-state">
        <div className="empty-icon">🖼️</div>
        <h3>Results appear here</h3>
        <p>
          Upload an image and click Analyze to
          see the verdict and Grad-CAM heatmap.
        </p>
      </div>
    );
  }

  if (error) {
    return (
      <div
        className="glass-card result-panel error-container"
        role="alert"
      >
        <div className="error-message">
          <div className="error-icon">⚠️</div>

          <div>
            <h3>Error</h3>
            <p>{error}</p>
          </div>
        </div>
      </div>
    );
  }

  const isFake = result?.label === "FAKE";

  return (
    <div className="glass-card result-panel">

      <div className="result-header">
        <h2>Analysis Result</h2>

        <span
          className={
            isFake
              ? "prediction-badge fake"
              : "prediction-badge real"
          }
        >
          {result.label}
        </span>
      </div>

      <div className="metrics-grid">

        <div className="metric-card">
          <span>Confidence</span>
          <h3>
            {formatPercentage(result.confidence)}
          </h3>
        </div>

        <div className="metric-card">
          <span>AI Probability</span>
          <h3>
            {formatPercentage(result.prob_fake)}
          </h3>
        </div>

        <div className="metric-card">
          <span>Real Probability</span>
          <h3>
            {formatPercentage(result.prob_real)}
          </h3>
        </div>

        <div className="metric-card">
          <span>Prediction</span>
          <h3>{result.label}</h3>
        </div>

      </div>

      <div className="confidence-bar-wrapper">
        <div
          className="confidence-bar"
          style={{
            width: `${result.confidence * 100}%`,
          }}
        />
      </div>

      <div className="cam-section">

        <div className="cam-header">
          <h3>Grad-CAM Heatmap</h3>

          <p>
            Highlighted regions indicate
            areas used most heavily by
            the model during prediction.
          </p>
        </div>

        {result.cam_image_base64 ? (
          <div className="cam-card">
            <img
              src={result.cam_image_base64}
              alt="Grad-CAM"
              className="cam-image"
            />
          </div>
        ) : (
          <div className="cam-unavailable">
            Heatmap unavailable
          </div>
        )}

      </div>

    </div>
  );
}