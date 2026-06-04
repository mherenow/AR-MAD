/**
 * UploadForm component
 *
 * Provides a file picker for image selection and a submit button to trigger
 * classification. Handles client-side validation display and loading state.
 *
 * Requirements: 1.1, 1.4, 1.5
 */

import React, { useState, useEffect } from "react";
import { validateFile } from "../utils/validation";
import Spinner from "./Spinner";

export interface UploadFormProps {
  /** Called when the user selects a file (or null when selection is cleared). */
  onFileSelect: (file: File | null) => void;
  /** Called when the user clicks the "Classify Image" submit button. */
  onSubmit: () => Promise<void>;
  /** The currently selected file, or null if none. */
  selectedFile: File | null;
  /** When true, a classification request is in-flight. */
  loading: boolean;
  /** Validation or API error message to display, or null if none. */
  error: string | null;
}

/**
 * UploadForm renders a file input restricted to supported image types and a
 * submit button labelled "Classify Image".
 *
 * The submit button is disabled whenever no valid file is selected or a
 * request is already in progress (Requirements 1.5, 1.6).
 */
const UploadForm: React.FC<UploadFormProps> = ({
  onFileSelect,
  onSubmit,
  selectedFile,
  loading,
  error,
}) => {
  // Local state for validation errors (Requirements 1.2, 1.3)
  const [validationError, setValidationError] = useState<string | null>(null);
  
  // Local state for thumbnail preview URL (Requirement 1.4)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  // Create and cleanup object URL for thumbnail preview (Requirement 1.4)
  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }

    // Create a new object URL for the valid selected file
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);

    // Cleanup function: revoke the object URL on unmount or when selectedFile changes
    return () => {
      URL.revokeObjectURL(objectUrl);
    };
  }, [selectedFile]); // Re-run effect when selectedFile changes

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    
    // Clear previous validation errors when new file selected (Requirement 1.2)
    setValidationError(null);
    
    if (file) {
      // Validate the selected file (Requirements 1.2, 1.3)
      const errorMessage = validateFile(file);
      
      if (errorMessage) {
        // Validation failed - display error and don't submit file
        setValidationError(errorMessage);
        onFileSelect(null);
      } else {
        // Validation passed - file is valid
        onFileSelect(file);
      }
    } else {
      // No file selected
      onFileSelect(null);
    }
  };

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    onSubmit();
  };

  /** Submit is disabled when there is no selected file or when loading. */
  const isSubmitDisabled = selectedFile === null || loading;

  // Display validation error or API error (Requirements 1.2, 1.3, 4.6)
  const displayError = validationError || error;

  return (
    <form onSubmit={handleSubmit} aria-label="Image upload form">
      {/* File input — accepts JPEG, PNG, BMP, and WebP (Requirements 1.1, 1.2) */}
      <div>
        <label htmlFor="image-input">Select an image to classify:</label>
        <input
          id="image-input"
          type="file"
          accept="image/jpeg,image/png,image/bmp,image/webp"
          onChange={handleFileChange}
          disabled={loading}
          aria-describedby={error ? "upload-error" : undefined}
        />
      </div>

      {/* Validation / error message (Requirements 1.2, 1.3) */}
      {displayError && (
        <p id="upload-error" role="alert" style={{ color: "red" }}>
          {displayError}
        </p>
      )}

      {/* Thumbnail preview (Requirement 1.4) */}
      {previewUrl && selectedFile && (
        <div style={{ marginTop: "1rem", marginBottom: "1rem" }}>
          <img
            src={previewUrl}
            alt="Preview of selected image"
            style={{ maxWidth: "300px", maxHeight: "300px", display: "block" }}
          />
        </div>
      )}

      {/* Loading spinner (Requirement 1.6) */}
      {loading && <Spinner label="Classifying image..." />}

      {/* Submit button (Requirements 1.5) */}
      <button type="submit" disabled={isSubmitDisabled}>
        Classify Image
      </button>
    </form>
  );
};

export default UploadForm;
