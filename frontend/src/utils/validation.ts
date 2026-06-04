/**
 * Client-side file validation utilities
 *
 * Validates Requirements 1.2 (MIME type) and 1.3 (file size)
 */

/**
 * Supported image MIME types for the classifier.
 * Any file whose type is not in this list will be rejected client-side.
 */
export const SUPPORTED_MIME_TYPES = [
  "image/jpeg",
  "image/png",
  "image/bmp",
  "image/webp",
] as const;

export type SupportedMimeType = (typeof SUPPORTED_MIME_TYPES)[number];

/** Maximum allowed file size: 10 MB */
export const MAX_FILE_SIZE = 10 * 1024 * 1024;

/**
 * Validates a File object for upload to the classifier.
 *
 * Checks performed (in order):
 *  1. MIME type must be one of the SUPPORTED_MIME_TYPES (Requirement 1.2)
 *  2. File size must not be 0 bytes (Requirement 1.3)
 *  3. File size must not exceed 10 MB (Requirement 1.3)
 *
 * @param file - The File object selected by the user
 * @returns An error message string if validation fails, or null if the file is valid
 */
export function validateFile(file: File): string | null {
  if (!SUPPORTED_MIME_TYPES.includes(file.type as SupportedMimeType)) {
    return `Unsupported file type: ${file.type || "(unknown)"}. Please select a JPEG, PNG, BMP, or WebP image.`;
  }

  if (file.size === 0) {
    return "File is empty. Please select a non-empty image file.";
  }

  if (file.size > MAX_FILE_SIZE) {
    const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
    return `File size (${sizeMB} MB) exceeds the 10 MB limit. Please select a smaller image.`;
  }

  return null;
}
