// frontend/src/App.jsx
import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css'; 
import AnalysisSummary from './AnalysisSummary';
import { useAuth } from './AuthContext'; 

function App() {
  const { logout } = useAuth();
  const [isLoading, setIsLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [originalFile, setOriginalFile] = useState(null);
  const [originalImage, setOriginalImage] = useState(null);
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [isBotThinking, setIsBotThinking] = useState(false);
  const [userRole, setUserRole] = useState('novice');
  const chatEndRef = useRef(null);

  const [isStressTesting, setIsStressTesting] = useState(false);
  const [stressTestResult, setStressTestResult] = useState(null);
  const [isGeneratingCF, setIsGeneratingCF] = useState(false);
  const [cfResult, setCfResult] = useState(null);
  const [cfError, setCfError] = useState(null);

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [chatMessages]);

  const fileToBase64 = (file) => new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result);
    reader.onerror = (error) => reject(error);
  });

  const handleImageSelect = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    setOriginalFile(file);
    setOriginalImage(URL.createObjectURL(file));
    setShowResults(false);
    setAnalysisResult(null);
    setStressTestResult(null);
    setCfResult(null); setCfError(null);
    setChatMessages([]);
    event.target.value = null;
  };
  
  const handleAnalyzeClick = async () => {
    if (!originalFile) return;
    setIsLoading(true);
    const formData = new FormData();
    formData.append('image', originalFile);
    formData.append('mode', userRole);
    try {
      const response = await axios.post('http://localhost:3000/api/detect', formData, { timeout: 300000 });
      setAnalysisResult(response.data);
      setChatMessages([{ from: 'bot', type: 'text', content: `Analysis complete. Verdict: ${response.data.verdict}.` }]);
      setShowResults(true);
    } catch (error) { alert("Error: Could not analyze the image."); } 
    finally { setIsLoading(false); }
  };

  const handleDownloadReport = async () => {
    if (!analysisResult) return;
    try {
      let originalBase64 = originalFile ? await fileToBase64(originalFile) : null;
      await axios.post('http://localhost:3000/api/report', {
        verdict: analysisResult.verdict,
        confidence: analysisResult.confidence,
        reason: analysisResult.reason,
        fidelity: analysisResult.fidelity,
        timestamp: new Date().toISOString(),
        originalImage: originalBase64, 
        heatmapImage: analysisResult.heatmap_image,
        robustness: stressTestResult,
        counterfactual: cfResult 
      }, { responseType: 'blob' }).then((response) => {
         const url = window.URL.createObjectURL(new Blob([response.data]));
         const link = document.createElement('a');
         link.href = url;
         link.setAttribute('download', `Report_${Date.now()}.pdf`);
         document.body.appendChild(link);
         link.click();
         link.remove();
      });
    } catch (error) { alert("Could not generate report."); }
  };

  const handleBackToUpload = () => {
    setOriginalFile(null); setOriginalImage(null); setShowResults(false); setAnalysisResult(null);
  };

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatInput.trim() || isBotThinking) return;
    const q = chatInput;
    setChatMessages(prev => [...prev, { from: 'user', type: 'text', content: q }]);
    setChatInput('');
    setIsBotThinking(true);
    try {
      const response = await axios.post('http://localhost:3000/api/chat', {
        question: q,
        analysisContext: { verdict: analysisResult.verdict, confidence: analysisResult.confidence },
        role: userRole
      });
      setChatMessages(prev => [...prev, { from: 'bot', type: 'text', content: response.data.reply }]);
    } catch (e) { setChatMessages(prev => [...prev, { from: 'bot', type: 'text', content: "Connection error." }]); } 
    finally { setIsBotThinking(false); }
  };

  const handleStressTest = async () => {
    if (!originalFile) return;
    setIsStressTesting(true);
    const formData = new FormData();
    formData.append('image', originalFile);
    try {
      const res = await axios.post('http://localhost:3000/api/stress-test', formData);
      setStressTestResult(res.data);
    } catch (e) { alert("Stress test failed."); }
    finally { setIsStressTesting(false); }
  };

  const handleGenerateCF = async () => {
    if (!originalFile) return;
    setIsGeneratingCF(true);
    setCfResult(null); setCfError(null);
    const formData = new FormData();
    formData.append('image', originalFile);
    try {
      const res = await axios.post('http://localhost:3000/api/counterfactual', formData, { timeout: 600000 });
      setCfResult(res.data);
    } catch (e) { 
        console.error(e);
        setCfError("Failed to generate."); 
    } finally { setIsGeneratingCF(false); }
  };

  // --- RENDER FUNCTIONS ---

  const renderMainContent = () => {
    if (isLoading) return <div className="loader"></div>;
    
    if (showResults && analysisResult) {
      return (
        <div className="dashboard-layout">
          <div className="dashboard-top-row">
            <div className="dashboard-col-left">
               <div className="image-display-card">
                 <h4>Original Image</h4>
                 <img src={originalImage} alt="Uploaded" />
               </div>
               <div className="verdict-section">
                 <div className="mode-indicator-container">
                    <span className={`mode-status-badge ${userRole}`}>
                      Running Mode: <strong>{userRole === 'novice' ? '⚡ Novice' : '🔬 Expert'}</strong>
                    </span>
                 </div>
                 <AnalysisSummary
                    verdict={analysisResult.verdict}
                    confidence={analysisResult.confidence}
                    summaryText={analysisResult.reason}
                    fidelity={analysisResult.fidelity}
                  />
               </div>
            </div>
            <div className="dashboard-col-right">
               <div className="image-display-card">
                 <h4>Analysis Heatmap</h4>
                 <img src={analysisResult.heatmap_image} alt="Heatmap" />
               </div>
               <div className="chat-card">
                  <div className="chat-header"><h4>AI Assistant Chat</h4></div>
                  <div className="live-chat-container">
                      <div className="live-chat-messages">
                        {chatMessages.map((msg, i) => (
                          <div key={i} className={`chat-bubble ${msg.from}`}>{msg.content}</div>
                        ))}
                        {isBotThinking && <div className="chat-bubble bot">...</div>}
                        <div ref={chatEndRef} />
                      </div>
                      <form className="live-chat-input-form" onSubmit={handleChatSubmit}>
                        <input type="text" className="live-chat-input" value={chatInput} onChange={(e) => setChatInput(e.target.value)} />
                        <button type="submit" className="live-chat-send-btn">➡️</button>
                      </form>
                   </div>
               </div>
            </div>
          </div>

          <div className="dashboard-bottom-stacked">
             <div className="tool-card">
                <div className="tool-header">
                    <h4>Robustness Stress Test</h4>
                    <button onClick={handleStressTest} disabled={isStressTesting} className="stress-test-btn">
                      {isStressTesting ? "Running..." : "▶ Run Test"}
                    </button>
                </div>
                {stressTestResult && (
                  <ul className="stress-list">
                    {Object.entries(stressTestResult).filter(([k])=>k!=='original').map(([k,v]) => (
                      <li key={k} className={v.verdict!==stressTestResult.original.verdict?'flipped':'stable'}>
                        {k}: {v.verdict} ({v.confidence}%)
                      </li>
                    ))}
                  </ul>
                )}
             </div>

             {/* --- INLINE COUNTERFACTUAL RESULTS --- */}
             <div className="tool-card">
                <div className="tool-header">
                    <h4>Counterfactual Generation</h4>
                    <button onClick={handleGenerateCF} disabled={isGeneratingCF || userRole !== 'expert'} className="cf-btn">
                      {isGeneratingCF ? "Generating..." : "🧬 Generate"}
                    </button>
                </div>
                {userRole !== 'expert' && <p className="cf-locked">Switch to Expert Mode to use this.</p>}
                
                {isGeneratingCF && <div className="modal-loading"><div className="loader-small"></div><p>Analyzing pixel perturbations...</p></div>}
                {cfError && <div className="modal-error"><p>Error: {cfError}</p></div>}

                {cfResult && (
                  <div className="cf-inline-results">
                    <div className="cf-text-box">
                        <strong>Detailed Modifications:</strong>
                        <pre>{cfResult.difference_text}</pre>
                    </div>
                    <div className="cf-images-row">
                      <div><img src={cfResult.original_image} alt="Orig" /><span>Original ({cfResult.original_pred_text})</span></div>
                      <div className="arrow">➜</div>
                      <div><img src={cfResult.counterfactual_image} alt="CF" /><span>Counterfactual ({cfResult.target_pred_text})</span></div>
                    </div>
                  </div>
                )}
             </div>
          </div>
        </div>
      );
    }
    
    if (originalImage) {
      return (
        <div className="preview-box">
          <h2>Image Preview</h2>
          <img src={originalImage} alt="Preview" className="preview-image" />
          <div className="preview-actions">
            <button onClick={handleBackToUpload} className="preview-btn cancel">Cancel</button>
            <button onClick={handleAnalyzeClick} className="preview-btn analyze">Analyze Image</button>
          </div>
        </div>
      );
    }

    return (
      <div className="upload-box">
        <div className="upload-role-section">
            <label className="role-label">Select Analysis Mode:</label>
            <div className="role-selector-large">
              <button className={`role-btn-large ${userRole === 'novice' ? 'active' : ''}`} onClick={() => setUserRole('novice')}>Novice</button>
              <button className={`role-btn-large ${userRole === 'expert' ? 'active' : ''}`} onClick={() => setUserRole('expert')}>Expert</button>
            </div>
            <p className="role-explainer">
              {userRole === 'novice' ? "⚡ Fast Result: Instant heatmap (0.5s)." : "🔬 Expert Result: Deep forensic scanning (30s)."}
            </p>
        </div>
        <div className="divider"></div>
        <div className="upload-dashed-area">
            <input type="file" id="file-upload" onChange={handleImageSelect} accept="image/png, image/jpeg" />
            <div className="upload-icon">⬆️</div>
            <h3>Upload Face Image</h3>
            <p>Upload a single face image to analyze for deepfake indicators.</p>
            <label htmlFor="file-upload" className="upload-btn-main">Select Image</label>
        </div>
      </div>
    );
  };

  return (
    <div className="app-root">
      <nav className="app-navbar">
        <div className="navbar-content">
          <div className="navbar-title">Interactive Deepfake Detector</div>
          <div className="navbar-controls">
            {showResults && !isLoading && (
              <>
                <button onClick={handleBackToUpload} className="upload-new-btn">Upload New</button>
                <button onClick={handleDownloadReport} className="download-btn" style={{marginRight: '10px'}}>📄 Report</button>
              </>
            )}
            <button onClick={logout} className="logout-btn">Logout</button>
          </div>
        </div>
      </nav>
      <div className="app-container">
        <div className="content-area">{renderMainContent()}</div>
      </div>
    </div>
  );
}

export default App;