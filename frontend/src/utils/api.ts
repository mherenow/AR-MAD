/**
 * API configuration and client utilities
 *
 * API_BASE_URL is read from the VITE_API_BASE_URL environment variable.
 * When not set, defaults to empty string to use relative URLs (for Vite dev proxy).
 * When set, uses the provided value as the base URL for all API requests.
 *
 * Validates Requirements 6.3, 6.4
 */

import type { ClassificationResult } from "../types/api";

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

/**
 * Submit an image file for AI/real classification.
 *
 * Builds a multipart/form-data request with the file attached under the
 * field name "image" (matching the backend's expected field name) and POSTs
 * it to `/classify`.
 *
 * On a non-200 response, the backend returns a JSON body with a `detail`
 * field (FastAPI error shape). This function extracts that message and
 * throws an Error so callers can display it directly to the user.
 *
 * Validates Requirements 2.1, 4.6
 *
 * @param file - The image File selected by the user.
 * @returns Parsed ClassificationResult on HTTP 200.
 * @throws Error with the backend's `detail` message on non-200 responses,
 *         or a generic message if the response body cannot be parsed.
 */
export async function classifyImage(file: File): Promise<ClassificationResult> {
  const formData = new FormData();
  formData.append("image", file);

  const response = await fetch(`${API_BASE_URL}/classify`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    // Attempt to parse the backend's JSON error body for a user-facing message.
    let detail: string;
    try {
      const errorData = await response.json();
      detail = errorData.detail || `HTTP ${response.status}`;
    } catch {
      detail = `HTTP ${response.status}`;
    }
    throw new Error(detail);
  }

  return (await response.json()) as ClassificationResult;
}

/**
 * Check whether the backend is healthy and its model is loaded.
 *
 * GETs `/health` and returns the parsed JSON body.
 * Throws an Error if the response is non-200 (e.g., 503 while model loads).
 *
 * Validates Requirements 5.3, 5.5
 *
 * @returns Object with a `status` field ("ok" or "loading").
 * @throws Error with a message describing the failure.
 */
export async function checkHealth(): Promise<{ status: string }> {
  const response = await fetch(`${API_BASE_URL}/health`);

  if (!response.ok) {
    throw new Error(`Health check failed: HTTP ${response.status}`);
  }

  return (await response.json()) as { status: string };
}
