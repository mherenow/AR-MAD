/**
 * Unit tests for UploadForm component
 * Tests file selection and validation logic (Requirements 1.2, 1.3)
 */

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import UploadForm from "./UploadForm";

describe("UploadForm - File Selection and Validation", () => {
  const mockOnFileSelect = vi.fn();
  const mockOnSubmit = vi.fn();

  const defaultProps = {
    onFileSelect: mockOnFileSelect,
    onSubmit: mockOnSubmit,
    selectedFile: null,
    loading: false,
    error: null,
  };

  it("should call onFileSelect with null for unsupported MIME type", () => {
    render(<UploadForm {...defaultProps} />);
    const fileInput = screen.getByLabelText(/select an image to classify/i);

    // Create a file with unsupported MIME type
    const invalidFile = new File(["test"], "test.gif", { type: "image/gif" });

    fireEvent.change(fileInput, { target: { files: [invalidFile] } });

    // Should display validation error
    expect(screen.getByRole("alert")).toHaveTextContent(/unsupported file type/i);
    
    // Should call onFileSelect with null (invalid file)
    expect(mockOnFileSelect).toHaveBeenCalledWith(null);
  });

  it("should call onFileSelect with null for empty file", () => {
    render(<UploadForm {...defaultProps} />);
    const fileInput = screen.getByLabelText(/select an image to classify/i);

    // Create a 0-byte file
    const emptyFile = new File([], "empty.jpg", { type: "image/jpeg" });

    fireEvent.change(fileInput, { target: { files: [emptyFile] } });

    // Should display validation error
    expect(screen.getByRole("alert")).toHaveTextContent(/empty/i);
    
    // Should call onFileSelect with null (invalid file)
    expect(mockOnFileSelect).toHaveBeenCalledWith(null);
  });

  it("should call onFileSelect with null for file exceeding 10MB", () => {
    render(<UploadForm {...defaultProps} />);
    const fileInput = screen.getByLabelText(/select an image to classify/i);

    // Create a file larger than 10MB
    const largeFile = new File([new ArrayBuffer(11 * 1024 * 1024)], "large.jpg", {
      type: "image/jpeg",
    });

    fireEvent.change(fileInput, { target: { files: [largeFile] } });

    // Should display validation error
    expect(screen.getByRole("alert")).toHaveTextContent(/exceeds.*10.*MB/i);
    
    // Should call onFileSelect with null (invalid file)
    expect(mockOnFileSelect).toHaveBeenCalledWith(null);
  });

  it("should call onFileSelect with file for valid JPEG", () => {
    render(<UploadForm {...defaultProps} />);
    const fileInput = screen.getByLabelText(/select an image to classify/i);

    // Create a valid file
    const validFile = new File(["test content"], "test.jpg", { type: "image/jpeg" });

    fireEvent.change(fileInput, { target: { files: [validFile] } });

    // Should NOT display validation error
    expect(screen.queryByRole("alert")).toBeNull();
    
    // Should call onFileSelect with the valid file
    expect(mockOnFileSelect).toHaveBeenCalledWith(validFile);
  });

  it("should call onFileSelect with file for valid PNG", () => {
    render(<UploadForm {...defaultProps} />);
    const fileInput = screen.getByLabelText(/select an image to classify/i);

    // Create a valid PNG file
    const validFile = new File(["test content"], "test.png", { type: "image/png" });

    fireEvent.change(fileInput, { target: { files: [validFile] } });

    // Should NOT display validation error
    expect(screen.queryByRole("alert")).toBeNull();
    
    // Should call onFileSelect with the valid file
    expect(mockOnFileSelect).toHaveBeenCalledWith(validFile);
  });

  it("should clear previous validation errors when new file is selected", () => {
    render(<UploadForm {...defaultProps} />);
    const fileInput = screen.getByLabelText(/select an image to classify/i);

    // First, select an invalid file
    const invalidFile = new File(["test"], "test.gif", { type: "image/gif" });
    fireEvent.change(fileInput, { target: { files: [invalidFile] } });

    // Should display validation error
    expect(screen.getByRole("alert")).toHaveTextContent(/unsupported file type/i);

    // Now select a valid file
    const validFile = new File(["test content"], "test.jpg", { type: "image/jpeg" });
    fireEvent.change(fileInput, { target: { files: [validFile] } });

    // Previous validation error should be cleared
    expect(screen.queryByRole("alert")).toBeNull();
    
    // Should call onFileSelect with the valid file
    expect(mockOnFileSelect).toHaveBeenLastCalledWith(validFile);
  });
});

describe("UploadForm - Loading State and Spinner", () => {
  const mockOnFileSelect = vi.fn();
  const mockOnSubmit = vi.fn();

  const defaultProps = {
    onFileSelect: mockOnFileSelect,
    onSubmit: mockOnSubmit,
    selectedFile: null,
    loading: false,
    error: null,
  };

  it("should display spinner when loading is true", () => {
    const validFile = new File(["test content"], "test.jpg", { type: "image/jpeg" });
    
    render(<UploadForm {...defaultProps} selectedFile={validFile} loading={true} />);

    // Should display spinner with status role
    const spinner = screen.getByRole("status");
    expect(spinner).toBeDefined();
    expect(spinner).toHaveAttribute("aria-label", "Classifying image...");
  });

  it("should not display spinner when loading is false", () => {
    const validFile = new File(["test content"], "test.jpg", { type: "image/jpeg" });
    
    render(<UploadForm {...defaultProps} selectedFile={validFile} loading={false} />);

    // Should NOT display spinner
    expect(screen.queryByRole("status")).toBeNull();
  });

  it("should disable submit button when loading is true", () => {
    const validFile = new File(["test content"], "test.jpg", { type: "image/jpeg" });
    
    render(<UploadForm {...defaultProps} selectedFile={validFile} loading={true} />);

    const submitButton = screen.getByRole("button", { name: /classify image/i });
    expect(submitButton).toBeDisabled();
  });

  it("should disable file input when loading is true", () => {
    const validFile = new File(["test content"], "test.jpg", { type: "image/jpeg" });
    
    render(<UploadForm {...defaultProps} selectedFile={validFile} loading={true} />);

    const fileInput = screen.getByLabelText(/select an image to classify/i);
    expect(fileInput).toBeDisabled();
  });
});
