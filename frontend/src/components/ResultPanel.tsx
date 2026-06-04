/**
 * ResultPanel Component
 *
 * Displays the classification result returned by the backend, including:
 * - Label (FAKE/REAL) with color coding
 * - Confidence percentage
 * - Probability scores (prob_fake and prob_real)
 * - CAM heatmap overlay image
 *
 * Also handles error display:
 * - When error prop is present, displays error message only
 * - Clears all result fields when error is present
 *
 * Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6
 */

import type { ClassificationResult } from '../types/api';
import './ResultPanel.css';

interface ResultPanelProps {
  result: ClassificationResult | null;
  error: string | null;
}

/**
 * Format a probability (0.0 - 1.0) as a percentage string with 1 decimal place.
 * Example: 0.873 → "87.3%"
 *
 * Requirements: 4.2, 4.3
 */
function formatPercentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export default function ResultPanel({ result, error }: ResultPanelProps) {
  // Return null if nothing to display
  if (!result && !error) {
    return null;
  }

  // When error is present, display error message and do NOT render result fields
  // Requirements: 4.6
  if (error) {
    return (
      <div className="result-panel error-container" role="alert">
        <div className="error-message">
          <svg className="error-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <circle cx="12" cy="12" r="10" strokeWidth="2" />
            <path d="M12 8v4m0 4h.01" strokeWidth="2" strokeLinecap="round" />
          </svg>
          <div>
            <strong>Error</strong>
            <p>{error}</p>
          </div>
        </div>
      </div>
    );
  }

  // Display classification result
  // Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
  if (result) {
    const labelColor = result.label === 'FAKE' ? 'red' : 'green';

    return (
      <div className="result-panel">
        <div className="result-header">
          {/* Display label with color coding: red for FAKE, green for REAL
              Requirements: 4.1 */}
          <h2 className={`result-label ${labelColor}`} style={{ color: labelColor }}>
            Classification: {result.label}
          </h2>

          {/* Display confidence as percentage with 1 decimal place
              Requirements: 4.2 */}
          <p className="confidence">
            <span>Confidence: </span>
            <span>{formatPercentage(result.confidence)}</span>
          </p>
        </div>

        <div className="probabilities">
          {/* Display prob_fake and prob_real with labels, formatted as percentages
              Requirements: 4.3 */}
          <div className="probability-item">
            <span className="probability-label">Probability Fake: </span>
            <span className="probability-value">{formatPercentage(result.prob_fake)}</span>
          </div>
          <div className="probability-item">
            <span className="probability-label">Probability Real: </span>
            <span className="probability-value">{formatPercentage(result.prob_real)}</span>
          </div>
        </div>

        {/* Display CAM heatmap if available
            Requirements: 4.4, 4.5 */}
        <div className="cam-section">
          <h3>Grad-CAM Heatmap</h3>
          {result.cam_image_base64 && result.cam_image_base64.length > 0 ? (
            <img
              src={result.cam_image_base64}
              alt="Grad-CAM heatmap overlay showing areas of the image that influenced the classification"
              className="cam-image"
            />
          ) : (
            <p className="cam-unavailable">Heatmap unavailable</p>
          )}
        </div>
      </div>
    );
  }

  return null;
}
