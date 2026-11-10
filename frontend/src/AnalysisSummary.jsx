// frontend/src/AnalysisSummary.jsx
import React from 'react';
import { ShieldIcon, ShieldAlertIcon, CheckCircleIcon, AlertTriangleIcon, InfoIcon } from './components/Icons';
import './AnalysisSummary.css';

function AnalysisSummary({ verdict, confidence, summaryText, fidelity }) {
  const isFake = verdict && verdict.toUpperCase() === 'FAKE';
  const confidenceValue = typeof confidence === 'number' ? confidence : parseFloat(confidence) || 0;
  
  const normalizedFidelity = fidelity ? fidelity.toUpperCase() : 'N/A';
  let fidelityTone = 'neutral';
  if (normalizedFidelity === 'HIGH') fidelityTone = 'high';
  else if (normalizedFidelity === 'MEDIUM') fidelityTone = 'medium';
  else if (normalizedFidelity === 'LOW') fidelityTone = 'low';

  // Format summary text into clean paragraphs or structured items
  const formatSummary = (text) => {
    if (!text) return null;

    // Check if the text contains numbered points like "1. ", "2. ", "3. "
    const numberedPattern = /(\d+\.\s+[^1-9]+)/g;
    const matches = text.match(numberedPattern);

    if (matches && matches.length > 1) {
      // Split pre-text before the first numbered point
      const firstIndex = text.indexOf(matches[0]);
      const intro = firstIndex > 0 ? text.slice(0, firstIndex).trim() : null;

      // Find recommendation/conclusion after last item if any
      const lastMatch = matches[matches.length - 1];
      const afterIndex = text.indexOf(lastMatch) + lastMatch.length;
      const outro = afterIndex < text.length ? text.slice(afterIndex).trim() : null;

      return (
        <div className="summary-structured-content">
          {intro && <p className="summary-intro">{intro}</p>}
          <ul className="summary-points-list">
            {matches.map((item, idx) => {
              const cleaned = item.replace(/^\d+\.\s*/, '').trim();
              return (
                <li key={idx} className="summary-point-item">
                  <span className="point-bullet">›</span>
                  <span className="point-text">{cleaned}</span>
                </li>
              );
            })}
          </ul>
          {outro && <p className="summary-outro">{outro}</p>}
        </div>
      );
    }

    // Default: format by sentences or paragraphs
    return (
      <div className="summary-paragraphs">
        <p className="summary-text">{text}</p>
      </div>
    );
  };

  return (
    <div className={`summary-card ${isFake ? 'verdict-fake' : 'verdict-real'}`}>
      {/* Verdict & Metrics Header */}
      <div className="summary-header">
        <div className="verdict-tag-group">
          <div className={`verdict-pill ${isFake ? 'fake' : 'real'}`}>
            {isFake ? <ShieldAlertIcon size={18} /> : <ShieldIcon size={18} />}
            <span className="verdict-label">
              {isFake ? 'Synthetic Manipulation Detected' : 'Authentic Media Verified'}
            </span>
          </div>
          <span className="verdict-sublabel">Model Classification: <strong>{verdict}</strong></span>
        </div>

        <div className="confidence-metric-box">
          <span className="confidence-title">Confidence</span>
          <span className="confidence-value">{confidenceValue.toFixed(1)}%</span>
        </div>
      </div>

      {/* Calibrated Confidence Progress Meter */}
      <div className="meter-container">
        <div 
          className={`meter-bar ${isFake ? 'fill-fake' : 'fill-real'}`}
          style={{ width: `${Math.min(Math.max(confidenceValue, 0), 100)}%` }}
        />
      </div>

      {/* Findings Narrative */}
      <div className="findings-section">
        <div className="findings-header">
          <InfoIcon size={15} />
          <h4>Forensic Analysis Findings</h4>
        </div>
        <div className="findings-body">
          {formatSummary(summaryText)}
        </div>
      </div>

      {/* Explanation Fidelity Row */}
      <div className="fidelity-footer">
        <div className="fidelity-label-wrap">
          <span className="fidelity-label">Explanation Fidelity</span>
          <span className="fidelity-desc">Local accuracy of explainability attribution</span>
        </div>
        <span className={`fidelity-tag ${fidelityTone}`}>
          {normalizedFidelity}
        </span>
      </div>
    </div>
  );
}

export default AnalysisSummary;
