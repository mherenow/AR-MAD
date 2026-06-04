/**
 * Spinner component
 *
 * Displays an animated loading spinner with accessible attributes.
 * Used to indicate when a classification request is in progress.
 *
 * Requirements: 1.6
 */

import React from "react";
import "./Spinner.css";

export interface SpinnerProps {
  /** Optional additional CSS class name */
  className?: string;
  /** Accessible label for screen readers */
  label?: string;
}

/**
 * Spinner renders an animated circular loading indicator.
 * It includes appropriate ARIA attributes for accessibility.
 */
const Spinner: React.FC<SpinnerProps> = ({ 
  className = "", 
  label = "Loading..." 
}) => {
  return (
    <div 
      className={`spinner ${className}`}
      role="status"
      aria-label={label}
      aria-live="polite"
    >
      <div className="spinner-circle"></div>
      <span className="spinner-label">{label}</span>
    </div>
  );
};

export default Spinner;
