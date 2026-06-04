/**
 * Unit tests for ResultPanel component
 *
 * Tests verify:
 * - Null rendering when no result or error
 * - Label color styling (red for FAKE, green for REAL) (Requirement 4.1)
 * - Percentage formatting with 1 decimal place (Requirements 4.2, 4.3)
 * - CAM heatmap conditional rendering (Requirements 4.4, 4.5)
 * - Error display behavior (Requirement 4.6)
 */

import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import ResultPanel from "./ResultPanel";
import type { ClassificationResult } from "../types/api";

/** Minimal valid FAKE classification result for testing. */
const FAKE_RESULT: ClassificationResult = {
  label: "FAKE",
  confidence: 0.873,
  prob_fake: 0.873,
  prob_real: 0.127,
  logit: 1.234,
  cam_image_base64: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA",
};

/** Minimal valid REAL classification result for testing. */
const REAL_RESULT: ClassificationResult = {
  label: "REAL",
  confidence: 0.654,
  prob_fake: 0.346,
  prob_real: 0.654,
  logit: -0.567,
  cam_image_base64: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA",
};

describe("ResultPanel", () => {
  describe("Null rendering", () => {
    it("returns null when both result and error are null", () => {
      const { container } = render(<ResultPanel result={null} error={null} />);
      expect(container.firstChild).toBeNull();
    });
  });

  describe("Label display (Requirement 4.1)", () => {
    it("displays FAKE label with red color", () => {
      render(<ResultPanel result={FAKE_RESULT} error={null} />);
      const heading = screen.getByRole("heading", { name: /Classification: FAKE/ });
      expect(heading).toBeInTheDocument();
      // The component sets style color to "red" which renders as rgb(255, 0, 0) in jsdom
      expect(heading).toHaveStyle({ color: "rgb(255, 0, 0)" });
    });

    it("displays REAL label with green color", () => {
      render(<ResultPanel result={REAL_RESULT} error={null} />);
      const heading = screen.getByRole("heading", { name: /Classification: REAL/ });
      expect(heading).toBeInTheDocument();
      // The component sets style color to "green" which renders as rgb(0, 128, 0) in jsdom
      expect(heading).toHaveStyle({ color: "rgb(0, 128, 0)" });
    });
  });

  describe("Confidence and probability formatting (Requirements 4.2, 4.3)", () => {
    it("formats confidence as percentage with 1 decimal place", () => {
      render(<ResultPanel result={FAKE_RESULT} error={null} />);
      // 0.873 should appear at least once as "87.3%"
      const elements = screen.getAllByText(/87\.3%/);
      expect(elements.length).toBeGreaterThan(0);
    });

    it("formats prob_fake with distinct label as percentage with 1 decimal place", () => {
      render(<ResultPanel result={FAKE_RESULT} error={null} />);
      expect(screen.getByText(/Probability Fake:/)).toBeInTheDocument();
    });

    it("formats prob_real with distinct label as percentage with 1 decimal place", () => {
      render(<ResultPanel result={FAKE_RESULT} error={null} />);
      expect(screen.getByText(/Probability Real:/)).toBeInTheDocument();
      expect(screen.getByText(/12\.7%/)).toBeInTheDocument();
    });

    it("formats 100% confidence as 100.0%", () => {
      const result: ClassificationResult = {
        ...FAKE_RESULT,
        confidence: 1.0,
        prob_fake: 1.0,
        prob_real: 0.0,
      };
      render(<ResultPanel result={result} error={null} />);
      const elements = screen.getAllByText(/100\.0%/);
      expect(elements.length).toBeGreaterThan(0);
    });

    it("formats 0% confidence as 0.0%", () => {
      const result: ClassificationResult = {
        ...REAL_RESULT,
        confidence: 0.0,
        prob_real: 0.0,
      };
      render(<ResultPanel result={result} error={null} />);
      const zeroElements = screen.getAllByText(/0\.0%/);
      expect(zeroElements.length).toBeGreaterThan(0);
    });
  });

  describe("CAM heatmap display (Requirements 4.4, 4.5)", () => {
    it("renders img element when cam_image_base64 is non-empty", () => {
      render(<ResultPanel result={FAKE_RESULT} error={null} />);
      const img = screen.getByRole("img");
      expect(img).toBeInTheDocument();
      expect(img).toHaveAttribute("src", FAKE_RESULT.cam_image_base64);
    });

    it("displays 'Heatmap unavailable' when cam_image_base64 is empty", () => {
      const result: ClassificationResult = { ...FAKE_RESULT, cam_image_base64: "" };
      render(<ResultPanel result={result} error={null} />);
      expect(screen.getByText("Heatmap unavailable")).toBeInTheDocument();
      expect(screen.queryByRole("img")).not.toBeInTheDocument();
    });
  });

  describe("Error display (Requirement 4.6)", () => {
    it("displays error message when error prop is present", () => {
      render(<ResultPanel result={null} error="File size exceeds 10MB limit" />);
      expect(screen.getByText(/File size exceeds 10MB limit/)).toBeInTheDocument();
    });

    it("does not render result fields when error is present", () => {
      render(<ResultPanel result={FAKE_RESULT} error="Something went wrong" />);
      // Error message should be displayed
      expect(screen.getByText(/Something went wrong/)).toBeInTheDocument();
      // Result heading should not be displayed
      expect(screen.queryByRole("heading", { name: /Classification: FAKE/ })).not.toBeInTheDocument();
      // Confidence and probabilities should not be shown
      expect(screen.queryByText(/87\.3%/)).not.toBeInTheDocument();
    });

    it("clears previous result when error is displayed", () => {
      render(<ResultPanel result={FAKE_RESULT} error="API error" />);
      // Result label should not be displayed
      expect(screen.queryByRole("img")).not.toBeInTheDocument();
      // Error should be displayed
      expect(screen.getByText(/API error/)).toBeInTheDocument();
    });
  });
});
