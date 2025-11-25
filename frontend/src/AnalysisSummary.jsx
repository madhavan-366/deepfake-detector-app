// frontend/src/AnalysisSummary.jsx
import React from 'react';
import './AnalysisSummary.css';

// --- MODIFIED: Added 'fidelity' to the props ---
function AnalysisSummary({ verdict, confidence, summaryText, fidelity }) {
  
  const verdictClass = verdict.toLowerCase();

  // --- NEW: Logic to style the fidelity score ---
  const fidelityClass = fidelity ? fidelity.toLowerCase() : 'low';
  let fidelityText = fidelity || 'N/A';

  return (
    <div className="summary-container">
      <div className="summary-header">
        <span className={`verdict-badge ${verdictClass}`}>{verdict}</span>
        <span className="confidence-percent">{confidence.toFixed(1)}%</span>
      </div>
      
      <div className="confidence-bar-container">
        <div 
          className="confidence-bar-fill" 
          style={{ width: `${confidence}%` }}
        ></div>
      </div>
      
      <div className="explanation">
        <strong>Explanation</strong>
        <p>{summaryText}</p>
      </div>
      
      {/* --- NEW: Fidelity Score Display --- */}
      <div className="fidelity-check">
        <strong>Explanation Fidelity:</strong>
        <span className={`fidelity-badge ${fidelityClass}`}>{fidelityText}</span>
      </div>
      {/* --- END OF NEW --- */}
    </div>
  );
}

export default AnalysisSummary;