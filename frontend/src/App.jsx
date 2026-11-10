// frontend/src/App.jsx
import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import AnalysisSummary from './AnalysisSummary';
import { useAuth } from './AuthContext';
import { API_BASE_URL } from './config';
import {
  ShieldIcon,
  UploadIcon,
  FileTextIcon,
  SendIcon,
  PlayIcon,
  RefreshIcon,
  LogOutIcon,
  AlertTriangleIcon,
  CheckCircleIcon,
  CrosshairIcon,
  ZapIcon,
  MicroscopeIcon,
  ArrowRightIcon,
  SpinnerIcon,
  UserIcon,
  BotIcon,
  CpuIcon,
  InfoIcon,
  KeyIcon,
  XIcon
} from './components/Icons';

function App() {
  const { logout, user, changePassword } = useAuth();
  const [isLoading, setIsLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [originalFile, setOriginalFile] = useState(null);
  const [originalImage, setOriginalImage] = useState(null);
  const [imageMetadata, setImageMetadata] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  // Account & Change Password Modal State
  const [isAccountModalOpen, setIsAccountModalOpen] = useState(false);
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [changePwError, setChangePwError] = useState('');
  const [changePwSuccess, setChangePwSuccess] = useState('');
  const [isChangingPw, setIsChangingPw] = useState(false);

  const handleOpenAccountModal = () => {
    setCurrentPassword('');
    setNewPassword('');
    setConfirmPassword('');
    setChangePwError('');
    setChangePwSuccess('');
    setIsAccountModalOpen(true);
  };

  const handleCloseAccountModal = () => {
    setIsAccountModalOpen(false);
    setCurrentPassword('');
    setNewPassword('');
    setConfirmPassword('');
    setChangePwError('');
    setChangePwSuccess('');
  };

  const handleChangePasswordSubmit = async (e) => {
    e.preventDefault();
    setChangePwError('');
    setChangePwSuccess('');

    if (!currentPassword) {
      setChangePwError('Please enter your current password.');
      return;
    }
    if (!newPassword || newPassword.length < 6) {
      setChangePwError('New password must be at least 6 characters long.');
      return;
    }
    if (newPassword !== confirmPassword) {
      setChangePwError('New passwords do not match. Please re-enter.');
      return;
    }
    if (newPassword === currentPassword) {
      setChangePwError('New password cannot be the same as your current password.');
      return;
    }

    setIsChangingPw(true);
    try {
      const res = await changePassword(currentPassword, newPassword, confirmPassword);
      if (res.success) {
        setChangePwSuccess(res.message || 'Password updated successfully!');
        setCurrentPassword('');
        setNewPassword('');
        setConfirmPassword('');
      } else {
        setChangePwError(res.message || 'Failed to update password. Please check your credentials.');
      }
    } catch {
      setChangePwError('An unexpected network error occurred. Please try again.');
    } finally {
      setIsChangingPw(false);
    }
  };

  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [isBotThinking, setIsBotThinking] = useState(false);
  const [userRole, setUserRole] = useState('novice'); // 'novice' (Standard) or 'expert' (Expert)
  const chatEndRef = useRef(null);

  const [isStressTesting, setIsStressTesting] = useState(false);
  const [stressTestResult, setStressTestResult] = useState(null);
  const [stressTestError, setStressTestError] = useState(null);

  const [isGeneratingCF, setIsGeneratingCF] = useState(false);
  const [cfResult, setCfResult] = useState(null);
  const [cfError, setCfError] = useState(null);

  const [isDownloadingReport, setIsDownloadingReport] = useState(false);
  const [generalError, setGeneralError] = useState(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatMessages, isBotThinking]);

  const fileToBase64 = (file) =>
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result);
      reader.onerror = (error) => reject(error);
    });

  // Extract dimensions and format metadata when a file is selected
  const processSelectedFile = (file) => {
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setGeneralError('Please upload a valid image file in JPG, JPEG, or PNG format.');
      return;
    }

    setGeneralError(null);
    setOriginalFile(file);

    const objectUrl = URL.createObjectURL(file);
    setOriginalImage(objectUrl);

    // Read natural dimensions
    const img = new Image();
    img.onload = () => {
      setImageMetadata({
        name: file.name,
        size: (file.size / (1024 * 1024)).toFixed(2) + ' MB',
        dimensions: `${img.naturalWidth} × ${img.naturalHeight} px`,
        type: file.type || 'image/jpeg'
      });
    };
    img.src = objectUrl;

    setShowResults(false);
    setAnalysisResult(null);
    setStressTestResult(null);
    setStressTestError(null);
    setCfResult(null);
    setCfError(null);
    setChatMessages([]);
  };

  const handleImageSelect = (event) => {
    const file = event.target.files?.[0];
    if (file) processSelectedFile(file);
    event.target.value = null;
  };

  // Drag and drop handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) processSelectedFile(file);
  };

  const handleAnalyzeClick = async () => {
    if (!originalFile) return;
    setIsLoading(true);
    setGeneralError(null);

    const formData = new FormData();
    formData.append('image', originalFile);
    formData.append('mode', userRole);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/detect`, formData, { timeout: 300000 });
      setAnalysisResult(response.data);
      setChatMessages([
        {
          from: 'bot',
          type: 'text',
          content: `Analysis complete. Verdict: **${response.data.verdict}** (${response.data.confidence.toFixed(1)}% confidence). You can ask questions about the detection findings or explainability heatmaps below.`
        }
      ]);
      setShowResults(true);
    } catch (error) {
      console.error('Analysis error:', error);
      setGeneralError('Analysis failed. The detection service could not process this image.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownloadReport = async () => {
    if (!analysisResult || isDownloadingReport) return;
    setIsDownloadingReport(true);
    setGeneralError(null);

    try {
      const originalBase64 = originalFile ? await fileToBase64(originalFile) : null;
      const response = await axios.post(
        `${API_BASE_URL}/api/report`,
        {
          verdict: analysisResult.verdict,
          confidence: analysisResult.confidence,
          reason: analysisResult.reason,
          fidelity: analysisResult.fidelity,
          timestamp: new Date().toISOString(),
          originalImage: originalBase64,
          heatmapImage: analysisResult.heatmap_image,
          robustness: stressTestResult,
          counterfactual: cfResult,
          metadata: imageMetadata,
          userRole: userRole
        },
        { responseType: 'blob' }
      );

      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `Deepfake_Detector_Report_${Date.now()}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Report error:', error);
      setGeneralError('Could not generate the forensic PDF report. Please verify backend service.');
    } finally {
      setIsDownloadingReport(false);
    }
  };

  const handleBackToUpload = () => {
    setOriginalFile(null);
    setOriginalImage(null);
    setImageMetadata(null);
    setShowResults(false);
    setAnalysisResult(null);
    setStressTestResult(null);
    setStressTestError(null);
    setCfResult(null);
    setCfError(null);
    setChatMessages([]);
    setGeneralError(null);
  };

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatInput.trim() || isBotThinking) return;

    const question = chatInput.trim();
    setChatMessages((prev) => [...prev, { from: 'user', type: 'text', content: question }]);
    setChatInput('');
    setIsBotThinking(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/chat`, {
        question,
        analysisContext: {
          verdict: analysisResult?.verdict,
          confidence: analysisResult?.confidence
        },
        role: userRole
      });
      setChatMessages((prev) => [...prev, { from: 'bot', type: 'text', content: response.data.reply }]);
    } catch (err) {
      console.error('Chat error:', err);
      setChatMessages((prev) => [
        ...prev,
        {
          from: 'bot',
          type: 'text',
          content: 'Unable to communicate with the assistant service. Please try again.'
        }
      ]);
    } finally {
      setIsBotThinking(false);
    }
  };

  const handlePromptChipClick = (promptText) => {
    setChatInput(promptText);
  };

  const handleStressTest = async () => {
    if (!originalFile || isStressTesting) return;
    setIsStressTesting(true);
    setStressTestError(null);

    const formData = new FormData();
    formData.append('image', originalFile);

    try {
      const res = await axios.post(`${API_BASE_URL}/api/stress-test`, formData);
      setStressTestResult(res.data);
    } catch (e) {
      console.error('Stress test error:', e);
      setStressTestError('Robustness testing failed. The AI service encountered an error.');
    } finally {
      setIsStressTesting(false);
    }
  };

  const handleGenerateCF = async () => {
    if (!originalFile || isGeneratingCF || userRole !== 'expert') return;
    setIsGeneratingCF(true);
    setCfResult(null);
    setCfError(null);

    const formData = new FormData();
    formData.append('image', originalFile);

    try {
      const res = await axios.post(`${API_BASE_URL}/api/counterfactual`, formData, { timeout: 600000 });
      setCfResult(res.data);
    } catch (e) {
      console.error('Counterfactual error:', e);
      setCfError('Counterfactual generation failed. Model perturbation could not find an inversion.');
    } finally {
      setIsGeneratingCF(false);
    }
  };

  // Safe markdown-like renderer for Chat Assistant text
  const renderMessageContent = (content) => {
    if (!content) return null;
    const lines = content.split('\n');

    return lines.map((line, idx) => {
      const parts = line.split(/(\*\*[^*]+\*\*)/g);
      const renderedLine = parts.map((part, pIdx) => {
        if (part.startsWith('**') && part.endsWith('**')) {
          return <strong key={pIdx}>{part.slice(2, -2)}</strong>;
        }
        return part;
      });

      if (line.trim().startsWith('•') || line.trim().startsWith('*') || line.trim().startsWith('-')) {
        return (
          <div key={idx} className="chat-bullet-line">
            <span className="bullet-marker">›</span>
            <span>{renderedLine}</span>
          </div>
        );
      }

      return (
        <p key={idx} className="chat-line">
          {renderedLine}
        </p>
      );
    });
  };

  // Format counterfactual difference text into structured items
  const formatCfDifferences = (diffText) => {
    if (!diffText) return null;
    const items = diffText.split('\n').filter((l) => l.trim().length > 0);

    return (
      <div className="cf-diff-list">
        {items.map((item, index) => {
          const cleanItem = item.replace(/^[•\s*-]+/, '').trim();
          return (
            <div key={index} className="cf-diff-chip">
              <span className="cf-diff-dot" />
              <span className="cf-diff-text">{cleanItem}</span>
            </div>
          );
        })}
      </div>
    );
  };

  // ----------------------------------------------------
  // MAIN CONTENT RENDER LOGIC
  // ----------------------------------------------------

  const renderMainContent = () => {
    // 1. LOADING STATE
    if (isLoading) {
      return (
        <div className="loading-card">
          <div className="loading-preview-wrapper">
            {originalImage && (
              <div className="loading-canvas">
                <img src={originalImage} alt="Scanning target" className="loading-image" />
                <div className="scanning-laser" />
              </div>
            )}
          </div>
          <div className="loading-telemetry">
            <div className="loading-spinner-wrap">
              <SpinnerIcon size={24} className="loading-spinner-icon" />
            </div>
            <h3 className="loading-title">Neural Image Analysis in Progress</h3>
            <p className="loading-desc">
              {userRole === 'expert'
                ? 'Aligning face landmarks, computing dual-stream consensus prediction, and sampling 50 LIME perturbation attributions (~30s)...'
                : 'Extracting face landmarks, running convolutional inference, and generating Grad-CAM activation heatmap (~0.5s)...'}
            </p>
            <div className="loading-bar-track">
              <div className="loading-bar-indeterminate" />
            </div>
            <div className="loading-meta-info">
              <span>Target: {originalFile?.name || 'Selected Media'}</span>
              <span>Mode: {userRole === 'expert' ? 'Expert Forensic' : 'Standard'}</span>
            </div>
          </div>
        </div>
      );
    }

    // 2. RESULTS DASHBOARD
    if (showResults && analysisResult) {
      const isFake = analysisResult.verdict?.toUpperCase() === 'FAKE';

      return (
        <div className="dashboard-layout">
          {/* Top Status & Context Bar */}
          <div className="dashboard-meta-bar">
            <div className="meta-left">
              <span className="meta-badge-tag">
                <CrosshairIcon size={14} />
                <span>Analysis Complete</span>
              </span>
              <span className="meta-divider">|</span>
              <span className="meta-mode-text">
                Mode:{' '}
                <strong className={userRole === 'expert' ? 'mode-expert' : 'mode-standard'}>
                  {userRole === 'expert' ? 'Expert (LIME Heatmap)' : 'Standard (Grad-CAM Heatmap)'}
                </strong>
              </span>
            </div>
            <div className="meta-right">
              <span className="meta-timestamp">
                Analyzed at {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              </span>
            </div>
          </div>

          {/* Top Diagnostic Grid */}
          <div className="dashboard-top-row">
            {/* Left Column: Input Portrait + Analysis Summary */}
            <div className="dashboard-col-left">
              <div className="evidence-panel">
                <div className="panel-header">
                  <div className="panel-title-group">
                    <CrosshairIcon size={16} />
                    <h4>Original Image (Cropped Face)</h4>
                  </div>
                  <span className="panel-tag">Input Media</span>
                </div>
                <div className="image-viewport">
                  <img src={originalImage} alt="Input facial crop" className="evidence-img" />
                </div>
                <div className="heatmap-caption">
                  <CrosshairIcon size={14} />
                  <span>
                    Extracted facial crop evaluated by the convolutional neural network.
                  </span>
                </div>
              </div>

              {/* Verdict Summary Card */}
              <AnalysisSummary
                verdict={analysisResult.verdict}
                confidence={analysisResult.confidence}
                summaryText={analysisResult.reason}
                fidelity={analysisResult.fidelity}
              />
            </div>

            {/* Right Column: Heatmap Viewport + Assistant */}
            <div className="dashboard-col-right">
              <div className="evidence-panel">
                <div className="panel-header">
                  <div className="panel-title-group">
                    <MicroscopeIcon size={16} />
                    <h4>
                      {userRole === 'expert'
                        ? 'Analysis Heatmap (LIME)'
                        : 'Analysis Heatmap (Grad-CAM)'}
                    </h4>
                  </div>
                  <span className={`panel-tag ${isFake ? 'tag-fake' : 'tag-real'}`}>
                    {isFake ? 'Tampering Indicators' : 'Natural Attributes'}
                  </span>
                </div>
                <div className="image-viewport">
                  <img
                    src={analysisResult.heatmap_image}
                    alt="Attribution heatmap"
                    className="evidence-img"
                  />
                </div>
                <div className="heatmap-caption">
                  <InfoIcon size={14} />
                  <span>
                    Highlighted warm regions represent the primary facial landmarks evaluated by the AI model.
                  </span>
                </div>
              </div>

              {/* AI Assistant Chat Card */}
              <div className="copilot-panel">
                <div className="copilot-header">
                  <div className="copilot-title-group">
                    <BotIcon size={18} />
                    <h4>AI Forensic Assistant</h4>
                  </div>
                  <span className="copilot-badge">Interactive Q&amp;A</span>
                </div>

                <div className="copilot-messages-container">
                  {chatMessages.map((msg, i) => (
                    <div key={i} className={`copilot-bubble ${msg.from}`}>
                      <div className="bubble-header">
                        {msg.from === 'bot' ? (
                          <>
                            <BotIcon size={14} />
                            <span>AI Assistant</span>
                          </>
                        ) : (
                          <>
                            <UserIcon size={14} />
                            <span>User</span>
                          </>
                        )}
                      </div>
                      <div className="bubble-content">{renderMessageContent(msg.content)}</div>
                    </div>
                  ))}

                  {isBotThinking && (
                    <div className="copilot-bubble bot thinking">
                      <div className="bubble-header">
                        <BotIcon size={14} />
                        <span>AI Assistant</span>
                      </div>
                      <div className="typing-indicator">
                        <span />
                        <span />
                        <span />
                      </div>
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>

                {/* Quick Suggestion Chips */}
                <div className="copilot-chips-row">
                  <button
                    type="button"
                    className="chip-btn"
                    onClick={() => handlePromptChipClick('What specific features in the heatmap influenced this verdict?')}
                  >
                    Explain heatmap
                  </button>
                  <button
                    type="button"
                    className="chip-btn"
                    onClick={() => handlePromptChipClick('How does explanation fidelity affect confidence?')}
                  >
                    Check fidelity
                  </button>
                  <button
                    type="button"
                    className="chip-btn"
                    onClick={() => handlePromptChipClick('Could compression or noise have triggered a false positive?')}
                  >
                    False positive risk
                  </button>
                </div>

                {/* Input Bar */}
                <form className="copilot-input-form" onSubmit={handleChatSubmit}>
                  <input
                    type="text"
                    className="copilot-input"
                    placeholder="Ask about prediction details, heatmap, or anomalies..."
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    disabled={isBotThinking}
                  />
                  <button
                    type="submit"
                    className="copilot-send-btn"
                    disabled={isBotThinking || !chatInput.trim()}
                    title="Send message"
                  >
                    <SendIcon size={16} />
                  </button>
                </form>
              </div>
            </div>
          </div>

          {/* Bottom Diagnostics Suite: Stress Test & Counterfactual Lab */}
          <div className="dashboard-bottom-section">
            {/* 1. Robustness Stress Suite */}
            <div className="diagnostic-tool-panel">
              <div className="tool-panel-header">
                <div className="tool-heading-wrap">
                  <ShieldIcon size={18} />
                  <div>
                    <h4>Robustness Stress Testing</h4>
                    <p>Evaluates prediction consistency against image degradations: blur, sensor noise, and JPEG compression.</p>
                  </div>
                </div>
                <button
                  onClick={handleStressTest}
                  disabled={isStressTesting}
                  className="tool-action-btn"
                >
                  {isStressTesting ? (
                    <>
                      <SpinnerIcon size={15} />
                      <span>Testing...</span>
                    </>
                  ) : (
                    <>
                      <PlayIcon size={13} />
                      <span>Run Stress Test</span>
                    </>
                  )}
                </button>
              </div>

              {stressTestError && (
                <div className="tool-alert error">
                  <AlertTriangleIcon size={16} />
                  <span>{stressTestError}</span>
                </div>
              )}

              {stressTestResult ? (
                <div className="diagnostic-table-wrapper">
                  <table className="diagnostic-table">
                    <thead>
                      <tr>
                        <th>Perturbation Test</th>
                        <th>Prediction</th>
                        <th>Confidence</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="table-row baseline">
                        <td>
                          <strong>Baseline (Original Upload)</strong>
                        </td>
                        <td>
                          <span className={`status-pill ${stressTestResult.original.verdict.toLowerCase()}`}>
                            {stressTestResult.original.verdict}
                          </span>
                        </td>
                        <td>
                          <span className="mono-num">{stressTestResult.original.confidence.toFixed(1)}%</span>
                        </td>
                        <td>
                          <span className="badge-neutral">Baseline Reference</span>
                        </td>
                      </tr>

                      {Object.entries(stressTestResult)
                        .filter(([k]) => k !== 'original')
                        .map(([key, item]) => {
                          const isFlipped = item.verdict !== stressTestResult.original.verdict;
                          const formattedKey =
                            key === 'blur'
                              ? 'Gaussian Blur (radius = 2)'
                              : key === 'noise'
                              ? 'Random Sensor Noise (σ = 0.05)'
                              : key === 'compression'
                              ? 'JPEG Compression (quality = 50)'
                              : key;

                          return (
                            <tr key={key} className={`table-row ${isFlipped ? 'row-flipped' : 'row-stable'}`}>
                              <td>{formattedKey}</td>
                              <td>
                                <span className={`status-pill ${item.verdict.toLowerCase()}`}>
                                  {item.verdict}
                                </span>
                              </td>
                              <td>
                                <span className="mono-num">{item.confidence.toFixed(1)}%</span>
                              </td>
                              <td>
                                {isFlipped ? (
                                  <span className="badge-vulnerable">
                                    <AlertTriangleIcon size={12} />
                                    <span>Prediction Flipped</span>
                                  </span>
                                ) : (
                                  <span className="badge-stable">
                                    <CheckCircleIcon size={12} />
                                    <span>Stable</span>
                                  </span>
                                )}
                              </td>
                            </tr>
                          );
                        })}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="tool-empty-prompt">
                  <InfoIcon size={16} />
                  <span>
                    Click <strong>Run Stress Test</strong> to check if noise or compression affects the model&apos;s output.
                  </span>
                </div>
              )}
            </div>

            {/* 2. Counterfactual Perturbation Lab */}
            <div className="diagnostic-tool-panel">
              <div className="tool-panel-header">
                <div className="tool-heading-wrap">
                  <CpuIcon size={18} />
                  <div>
                    <h4>Counterfactual Analysis</h4>
                    <p>Calculates minimal pixel modifications needed to reverse the prediction from real to fake or vice versa.</p>
                  </div>
                </div>

                <button
                  onClick={handleGenerateCF}
                  disabled={isGeneratingCF || userRole !== 'expert'}
                  className="tool-action-btn"
                >
                  {isGeneratingCF ? (
                    <>
                      <SpinnerIcon size={15} />
                      <span>Generating...</span>
                    </>
                  ) : (
                    <>
                      <CpuIcon size={14} />
                      <span>Generate Counterfactual</span>
                    </>
                  )}
                </button>
              </div>

              {userRole !== 'expert' && (
                <div className="tool-alert locked">
                  <InfoIcon size={16} />
                  <span>
                    Counterfactual analysis is only available in <strong>Expert Mode</strong>. Switch modes before analyzing to enable this simulation.
                  </span>
                </div>
              )}

              {isGeneratingCF && (
                <div className="tool-progress-box">
                  <SpinnerIcon size={20} className="tool-progress-spinner" />
                  <span>Computing pixel perturbations and generating counterfactual image...</span>
                </div>
              )}

              {cfError && (
                <div className="tool-alert error">
                  <AlertTriangleIcon size={16} />
                  <span>{cfError}</span>
                </div>
              )}

              {cfResult && (
                <div className="cf-results-layout">
                  <div className="cf-comparison-stage">
                    <div className="cf-image-card">
                      <div className="cf-image-header">
                        <span>Original State</span>
                        <span className={`cf-class-tag ${cfResult.original_pred_text.toLowerCase()}`}>
                          {cfResult.original_pred_text}
                        </span>
                      </div>
                      <div className="cf-img-box">
                        <img src={cfResult.original_image} alt="Original state" className="cf-preview-img" />
                      </div>
                    </div>

                    <div className="cf-transition-indicator">
                      <div className="cf-arrow-box">
                        <ArrowRightIcon size={20} className="cf-arrow-icon" />
                      </div>
                      <span className="cf-trans-label">Inverted</span>
                    </div>

                    <div className="cf-image-card">
                      <div className="cf-image-header">
                        <span>Modified State</span>
                        <span className={`cf-class-tag ${cfResult.target_pred_text.toLowerCase()}`}>
                          {cfResult.target_pred_text}
                        </span>
                      </div>
                      <div className="cf-img-box">
                        <img src={cfResult.counterfactual_image} alt="Counterfactual state" className="cf-preview-img" />
                      </div>
                    </div>
                  </div>

                  <div className="cf-details-card">
                    <div className="cf-details-title">
                      <InfoIcon size={14} />
                      <h5>Modifications Applied</h5>
                    </div>
                    {formatCfDifferences(cfResult.difference_text)}
                  </div>
                </div>
              )}
            </div>

            {/* 3. Responsible-Use Disclaimer Panel in Results */}
            <div className="disclaimer-panel">
              <div className="disclaimer-header">
                <AlertTriangleIcon size={18} className="disclaimer-icon" />
                <h4>Responsible-Use Disclaimer</h4>
              </div>
              <p className="disclaimer-text">
                This tool provides AI-generated predictions and should be used for informational purposes only. Results may be inaccurate and should not be treated as definitive proof that an image is real or manipulated. Always verify important findings using additional evidence.
              </p>
              <p className="disclaimer-subtext">
                Users should upload only images they have permission to use and avoid uploading sensitive or confidential images.
              </p>
            </div>
          </div>
        </div>
      );
    }

    // 3. IMAGE PREVIEW (Pre-Flight Verification)
    if (originalImage) {
      return (
        <div className="preview-container">
          <div className="preview-card">
            <div className="preview-header">
              <div className="preview-title-group">
                <CrosshairIcon size={20} />
                <h3>Image Preview &amp; Verification</h3>
              </div>
              <span className="preview-badge">Ready for Analysis</span>
            </div>

            <div className="preview-grid">
              {/* Image Inspection Box */}
              <div className="preview-canvas-wrap">
                <div className="canvas-crosshair top-left" />
                <div className="canvas-crosshair top-right" />
                <div className="canvas-crosshair bottom-left" />
                <div className="canvas-crosshair bottom-right" />
                <img src={originalImage} alt="Selected target preview" className="preview-img-element" />
              </div>

              {/* File & Pipeline Metadata */}
              <div className="preview-metadata-panel">
                <h4 className="meta-panel-title">Image Details</h4>

                <div className="meta-spec-list">
                  <div className="meta-spec-item">
                    <span className="spec-label">File Name</span>
                    <span className="spec-value mono-val">{imageMetadata?.name || originalFile?.name}</span>
                  </div>
                  <div className="meta-spec-item">
                    <span className="spec-label">File Size</span>
                    <span className="spec-value mono-val">{imageMetadata?.size || 'Pending'}</span>
                  </div>
                  <div className="meta-spec-item">
                    <span className="spec-label">Resolution</span>
                    <span className="spec-value mono-val">{imageMetadata?.dimensions || 'Detecting...'}</span>
                  </div>
                  <div className="meta-spec-item">
                    <span className="spec-label">Format</span>
                    <span className="spec-value mono-val">{imageMetadata?.type || 'image/jpeg'}</span>
                  </div>
                  <div className="meta-spec-item">
                    <span className="spec-label">Analysis Mode</span>
                    <span className="spec-value highlight">
                      {userRole === 'expert' ? 'Expert Forensic (LIME)' : 'Standard (Grad-CAM)'}
                    </span>
                  </div>
                  <div className="meta-spec-item">
                    <span className="spec-label">Processing</span>
                    <span className="spec-value">Automatic MTCNN Face Bounding Box</span>
                  </div>
                </div>

                {generalError && (
                  <div className="preview-error-alert">
                    <AlertTriangleIcon size={16} />
                    <span>{generalError}</span>
                  </div>
                )}

                <div className="preview-action-row">
                  <button type="button" onClick={handleBackToUpload} className="btn-secondary">
                    Change Image
                  </button>
                  <button type="button" onClick={handleAnalyzeClick} className="btn-primary">
                    <PlayIcon size={14} />
                    <span>Analyze Image</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      );
    }

    // 4. UPLOAD & INGESTION STATE (Initial View)
    return (
      <div className="upload-container">
        {/* Step 1: Mode Selection */}
        <div className="mode-selection-panel">
          <div className="section-label-row">
            <span className="step-tag">Step 1</span>
            <label className="section-title">Select Analysis Mode</label>
          </div>

          <div className="mode-cards-grid">
            {/* Standard Mode Card */}
            <div
              className={`mode-card ${userRole === 'novice' ? 'active' : ''}`}
              onClick={() => setUserRole('novice')}
            >
              <div className="mode-card-header">
                <div className="mode-icon-badge standard">
                  <ZapIcon size={20} />
                </div>
                <div className="mode-title-wrap">
                  <h4>Standard Scan</h4>
                  <span className="mode-subtitle">Fast Result (~0.5s)</span>
                </div>
                <div className={`mode-radio ${userRole === 'novice' ? 'selected' : ''}`} />
              </div>
              <ul className="mode-feature-list">
                <li>• Grad-CAM visual attention heatmap</li>
                <li>• Instant consensus classification</li>
                <li>• Clear, straightforward findings explanation</li>
              </ul>
            </div>

            {/* Expert Mode Card */}
            <div
              className={`mode-card ${userRole === 'expert' ? 'active' : ''}`}
              onClick={() => setUserRole('expert')}
            >
              <div className="mode-card-header">
                <div className="mode-icon-badge expert">
                  <MicroscopeIcon size={20} />
                </div>
                <div className="mode-title-wrap">
                  <h4>Expert Scan</h4>
                  <span className="mode-subtitle">Deep Scan (~30s)</span>
                </div>
                <div className={`mode-radio ${userRole === 'expert' ? 'selected' : ''}`} />
              </div>
              <ul className="mode-feature-list">
                <li>• LIME superpixel perturbation heatmap</li>
                <li>• In-depth artifact evaluation &amp; attribution</li>
                <li>• Unlocks Counterfactual modification simulation</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Step 2: Image Ingestion Dropzone */}
        <div className="ingestion-panel">
          <div className="section-label-row">
            <span className="step-tag">Step 2</span>
            <label className="section-title">Upload Face Image</label>
          </div>

          <div
            className={`dropzone-box ${isDragging ? 'dragging' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <input
              type="file"
              id="file-upload"
              onChange={handleImageSelect}
              accept="image/png, image/jpeg, image/jpg"
              style={{ display: 'none' }}
            />

            <div className="dropzone-inner">
              <div className="dropzone-icon-circle">
                <UploadIcon size={30} />
              </div>

              <h3 className="dropzone-heading">Drag and drop face image here</h3>
              <p className="dropzone-subheading">
                Upload a single facial image to inspect for synthetic manipulation, face-swapping, or deepfake artifacts.
              </p>

              <label htmlFor="file-upload" className="dropzone-browse-btn">
                <span>Select Image File</span>
              </label>

              <div className="dropzone-footer-specs">
                <span className="spec-badge">Formats: JPG, JPEG, PNG</span>
                <span className="spec-badge">Max 15 MB</span>
                <span className="spec-badge">Automatic Face Crop</span>
              </div>
            </div>
          </div>

          {generalError && (
            <div className="upload-error-alert">
              <AlertTriangleIcon size={16} />
              <span>{generalError}</span>
            </div>
          )}
        </div>

        {/* Professional "How to Use" Section */}
        <div className="how-to-use-panel">
          <div className="section-label-row">
            <span className="step-tag guide">Guide</span>
            <label className="section-title">How to Use Deepfake Detector</label>
          </div>

          <div className="how-to-use-grid">
            <div className="how-to-step-card">
              <div className="step-num-badge">1</div>
              <div className="step-content">
                <h5>Upload an Image</h5>
                <p>Choose an image in a supported format (JPG, JPEG, or PNG).</p>
              </div>
            </div>

            <div className="how-to-step-card">
              <div className="step-num-badge">2</div>
              <div className="step-content">
                <h5>Choose Mode</h5>
                <p>Select Standard for fast scanning or Expert for deep LIME analysis.</p>
              </div>
            </div>

            <div className="how-to-step-card">
              <div className="step-num-badge">3</div>
              <div className="step-content">
                <h5>Analyze Image</h5>
                <p>Click &quot;Analyze Image&quot; and wait a few moments for the model to process.</p>
              </div>
            </div>

            <div className="how-to-step-card">
              <div className="step-num-badge">4</div>
              <div className="step-content">
                <h5>Review Results</h5>
                <p>Inspect the prediction verdict, confidence score, and visual heatmap.</p>
              </div>
            </div>

            <div className="how-to-step-card">
              <div className="step-num-badge">5</div>
              <div className="step-content">
                <h5>Download Report</h5>
                <p>Export the complete findings as a verifiable PDF report.</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // ----------------------------------------------------
  // GLOBAL APPLICATION SHELL
  // ----------------------------------------------------
  return (
    <div className="app-root">
      {/* Top Navigation Bar */}
      <header className="app-navbar">
        <div className="navbar-content">
          <div className="navbar-brand">
            <div className="brand-logo-wrap">
              <ShieldIcon size={20} className="brand-logo-icon" />
            </div>
            <div className="brand-text-wrap">
              <span className="brand-title">Deepfake Detector</span>
              <span className="brand-tagline">AI Media Integrity</span>
            </div>
          </div>

          <div className="navbar-actions">
            {showResults && !isLoading && (
              <>
                <button
                  type="button"
                  onClick={handleBackToUpload}
                  className="nav-action-btn secondary"
                  title="Upload a new image for analysis"
                >
                  <RefreshIcon size={15} />
                  <span>New Scan</span>
                </button>
                <button
                  type="button"
                  onClick={handleDownloadReport}
                  className="nav-action-btn primary"
                  disabled={isDownloadingReport}
                  title="Download verification PDF report"
                >
                  {isDownloadingReport ? (
                    <>
                      <SpinnerIcon size={15} />
                      <span>Exporting PDF...</span>
                    </>
                  ) : (
                    <>
                      <FileTextIcon size={15} />
                      <span>Download Report</span>
                    </>
                  )}
                </button>
              </>
            )}

            <button
              type="button"
              onClick={handleOpenAccountModal}
              className="nav-action-btn secondary"
              title="Account Settings & Security"
            >
              <KeyIcon size={15} />
              <span>Account</span>
            </button>

            <button
              type="button"
              onClick={logout}
              className="nav-action-btn logout"
              title="Sign out of Deepfake Detector"
            >
              <LogOutIcon size={15} />
              <span>Sign Out</span>
            </button>
          </div>
        </div>
      </header>

      {/* Main Workspace Canvas */}
      <main className="app-container">
        <div className="content-area">{renderMainContent()}</div>
      </main>

      {/* Professional Application Footer */}
      <footer className="app-footer">
        <div className="footer-content">
          <div className="footer-top">
            <div className="footer-brand">
              <ShieldIcon size={16} className="footer-brand-icon" />
              <span className="footer-brand-name">Deepfake Detector</span>
              <span className="footer-brand-sep">•</span>
              <span className="footer-brand-tag">AI Media Analysis Platform</span>
            </div>
            <div className="footer-meta">
              <span>Dual-Stream Xception Architecture • MTCNN Bounding Box</span>
            </div>
          </div>

          <div className="footer-disclaimer-box">
            <p>
              <strong>Disclaimer:</strong> This tool provides AI-generated predictions and should be used for informational purposes only. Results may be inaccurate and should not be treated as definitive proof that an image is real or manipulated. Always verify important findings using additional evidence. Please upload only images you have permission to use and avoid uploading sensitive or confidential images.
            </p>
          </div>

          <div className="footer-bottom">
            <span>&copy; {new Date().getFullYear()} Deepfake Detector. All rights reserved.</span>
            <span>Informational &amp; Research Forensic Tool</span>
          </div>
        </div>
      </footer>

      {/* Account Settings & Change Password Modal */}
      {isAccountModalOpen && (
        <div className="modal-backdrop" onClick={handleCloseAccountModal}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title-group">
                <div className="modal-title-icon">
                  <KeyIcon size={18} />
                </div>
                <div>
                  <h3 className="modal-title">Account Settings</h3>
                  <p className="modal-subtitle">Security &amp; Password Management</p>
                </div>
              </div>
              <button
                type="button"
                className="modal-close-btn"
                onClick={handleCloseAccountModal}
                aria-label="Close modal"
              >
                <XIcon size={18} />
              </button>
            </div>

            <div className="account-info-box">
              <div className="account-info-main">
                <div className="account-avatar">
                  <UserIcon size={18} />
                </div>
                <div className="account-details">
                  <span className="account-email-label">Authenticated Account</span>
                  <span className="account-email-val">{user?.email || 'Active Operator'}</span>
                </div>
              </div>
              <span className="account-status-badge">
                <span className="status-dot-emerald" />
                Active Session
              </span>
            </div>

            <div className="account-section">
              <div className="section-title-wrap">
                <LockIcon size={15} className="section-icon-emerald" />
                <h4 className="account-section-heading">Change Password</h4>
              </div>
              <p className="account-section-desc">
                Enter your current password followed by your new password to update your account credentials.
              </p>

              {changePwError && (
                <div className="account-alert error">
                  <AlertTriangleIcon size={16} />
                  <span>{changePwError}</span>
                </div>
              )}

              {changePwSuccess && (
                <div className="account-alert success">
                  <CheckCircleIcon size={16} />
                  <span>{changePwSuccess}</span>
                </div>
              )}

              <form onSubmit={handleChangePasswordSubmit} className="change-pw-form">
                <div className="modal-input-group">
                  <label htmlFor="current-pw">Current Password</label>
                  <div className="modal-input-wrap">
                    <span className="modal-input-icon">
                      <LockIcon size={15} />
                    </span>
                    <input
                      id="current-pw"
                      type="password"
                      placeholder="Enter current password"
                      value={currentPassword}
                      onChange={(e) => setCurrentPassword(e.target.value)}
                      disabled={isChangingPw}
                      autoComplete="current-password"
                      required
                    />
                  </div>
                </div>

                <div className="modal-input-group">
                  <label htmlFor="new-pw">New Password</label>
                  <div className="modal-input-wrap">
                    <span className="modal-input-icon">
                      <LockIcon size={15} />
                    </span>
                    <input
                      id="new-pw"
                      type="password"
                      placeholder="Minimum 6 characters"
                      value={newPassword}
                      onChange={(e) => setNewPassword(e.target.value)}
                      disabled={isChangingPw}
                      autoComplete="new-password"
                      required
                    />
                  </div>
                </div>

                <div className="modal-input-group">
                  <label htmlFor="confirm-pw">Confirm New Password</label>
                  <div className="modal-input-wrap">
                    <span className="modal-input-icon">
                      <LockIcon size={15} />
                    </span>
                    <input
                      id="confirm-pw"
                      type="password"
                      placeholder="Re-enter new password"
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      disabled={isChangingPw}
                      autoComplete="new-password"
                      required
                    />
                  </div>
                </div>

                <div className="modal-actions-row">
                  <button
                    type="button"
                    onClick={handleCloseAccountModal}
                    className="btn-modal-secondary"
                    disabled={isChangingPw}
                  >
                    Cancel
                  </button>
                  <button
                    type="submit"
                    className="btn-modal-primary"
                    disabled={isChangingPw}
                  >
                    {isChangingPw ? (
                      <>
                        <SpinnerIcon size={15} />
                        <span>Updating...</span>
                      </>
                    ) : (
                      <span>Update Password</span>
                    )}
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
