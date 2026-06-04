/**
 * TypeScript type definitions for the AI Image Classifier API.
 *
 * These types mirror the backend Pydantic models and define the shape
 * of data exchanged between the frontend and the FastAPI backend.
 */

/**
 * Classification label — "FAKE" when the model logit > 0.9244, "REAL" otherwise.
 *
 * Requirements: 3.1
 */
export type ClassificationLabel = "FAKE" | "REAL";

/**
 * The JSON response body returned by POST /classify on success (HTTP 200).
 *
 * Field constraints:
 *   - label:             "FAKE" | "REAL"  (Requirements 3.1)
 *   - confidence:        float in [0.0, 1.0]; equals prob_fake when label is "FAKE",
 *                        prob_real when label is "REAL"  (Requirements 3.2)
 *   - prob_fake:         float in [0.0, 1.0]  (Requirements 3.3)
 *   - prob_real:         float in [0.0, 1.0]; |prob_fake + prob_real - 1.0| ≤ 1e-6  (Requirements 3.3)
 *   - logit:             raw model output as a float  (Requirements 3.4)
 *   - cam_image_base64:  base64-encoded PNG string prefixed with "data:image/png;base64,"
 *                        suitable for direct use as an HTML <img> src attribute  (Requirements 3.6)
 */
export interface ClassificationResult {
  label: ClassificationLabel;
  confidence: number;
  prob_fake: number;
  prob_real: number;
  logit: number;
  cam_image_base64: string;
}

/**
 * The JSON response body returned by the backend on error responses.
 * Matches FastAPI's default error shape (HTTPException detail).
 */
export interface ApiError {
  detail: string;
}
