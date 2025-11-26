// main-backend/server.js
const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');
const path = require('path');
const cors = require('cors');
const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const PDFDocument = require('pdfkit'); 
const crypto = require('crypto');

const { GoogleGenerativeAI } = require('@google/generative-ai');
require('dotenv').config(); 

// Connect DB
mongoose.connect(process.env.MONGO_URI)
  .then(() => console.log('MongoDB connected successfully.'))
  .catch(err => console.error('MongoDB connection error:', err));

const User = require('./models/User');

const app = express();
const port = 3000;
app.use(cors());

// --- CRITICAL: Increase Payload Limit ---
app.use(express.json({ limit: '50mb' })); 
app.use(express.urlencoded({ limit: '50mb', extended: true }));

const uploadDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadDir)) { fs.mkdirSync(uploadDir); }
const storage = multer.diskStorage({
  destination: (req, file, cb) => { cb(null, 'uploads/'); },
  filename: (req, file, cb) => { cb(null, Date.now() + '-' + file.originalname); },
});
const upload = multer({ storage: storage });

if (!process.env.GEMINI_API_KEY) console.error("CRITICAL ERROR: GEMINI_API_KEY missing");
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// --- AUTH ROUTES ---
app.post('/api/auth/register', async (req, res) => {
  const { email, password } = req.body;
  try {
    let user = await User.findOne({ email });
    if (user) return res.status(400).json({ msg: 'User already exists' });
    user = new User({ email, password });
    const salt = await bcrypt.genSalt(10);
    user.password = await bcrypt.hash(password, salt);
    await user.save();
    const payload = { user: { id: user.id } };
    jwt.sign(payload, process.env.JWT_SECRET, { expiresIn: '1h' }, (err, token) => {
      if (err) throw err;
      res.json({ token });
    });
  } catch (err) { res.status(500).send('Server error'); }
});

app.post('/api/auth/login', async (req, res) => {
  const { email, password } = req.body;
  try {
    let user = await User.findOne({ email });
    if (!user) return res.status(400).json({ msg: 'Invalid credentials' });
    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) return res.status(400).json({ msg: 'Invalid credentials' });
    const payload = { user: { id: user.id } };
    jwt.sign(payload, process.env.JWT_SECRET, { expiresIn: '1h' }, (err, token) => {
      if (err) throw err;
      res.json({ token });
    });
  } catch (err) { res.status(500).send('Server error'); }
});

// --- HELPER FOR URL ---
// This automatically picks the right URL (Docker vs Localhost)
const getAiUrl = (endpoint) => {
    const baseUrl = process.env.AI_SERVICE_URL || 'http://127.0.0.1:5000';
    return `${baseUrl}${endpoint}`;
};

// --- AI ROUTES ---
app.post('/api/detect', upload.single('image'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No image file' });
  
  // FIX 1: Use Dynamic URL
  const pythonApiUrl = getAiUrl('/api/detect');

  try {
    const formData = new FormData();
    formData.append('image', fs.createReadStream(req.file.path));
    if (req.body.mode) formData.append('mode', req.body.mode);
    console.log(`Sending request to Python API at: ${pythonApiUrl}`);
    const response = await axios.post(pythonApiUrl, formData, { headers: { ...formData.getHeaders() } });
    fs.unlinkSync(req.file.path);
    res.json(response.data);
  } catch (error) {
    if (fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
    console.error("AI Error:", error.message);
    res.status(500).json({ error: 'AI service failed' });
  }
});

app.post('/api/chat', async (req, res) => {
  try {
    const { question, analysisContext, role } = req.body;
    const userRole = role || 'novice';
    
    // --- FIX: CLEAN THE CONTEXT ---
    // We create a copy of the data and DELETE the huge image strings
    // This prevents the "Connection Error" / Payload too large issue
    const cleanContext = { ...analysisContext };
    delete cleanContext.heatmap_image;
    delete cleanContext.original_image;
    delete cleanContext.counterfactual_image; 

    // Use the stable model
    const modelName = "gemini-2.5-flash"; 
    const geminiModel = genAI.getGenerativeModel({ model: modelName });
    
    const prompt = `
      You are an expert AI Forensics Assistant.
      APP CONCEPTS: Heatmap, Fidelity, Robustness, Counterfactual.
      ANALYSIS CONTEXT: ${JSON.stringify(cleanContext, null, 2)}
      USER ROLE: ${userRole}
      USER QUESTION: "${question}"
      GUIDELINES: Answer directly and briefly. Do not mention that you cannot see the image (you have the forensic data).
    `;
    
    const result = await geminiModel.generateContent(prompt);
    const response = await result.response;
    const text = response.text();
    res.json({ reply: text });
  } catch (error) {
    console.error("Chat Error:", error); // This helps us see the real error in logs
    res.status(500).json({ reply: "Sorry, I'm having trouble connecting to my brain right now." });
  }
});

app.post('/api/stress-test', upload.single('image'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No image file' });
  
  // FIX 2: Use Dynamic URL
  const pythonApiUrl = getAiUrl('/api/stress-test');

  try {
    const formData = new FormData();
    formData.append('image', fs.createReadStream(req.file.path));
    const response = await axios.post(pythonApiUrl, formData, { headers: { ...formData.getHeaders() } });
    fs.unlinkSync(req.file.path);
    res.json(response.data);
  } catch (error) {
    if (fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
    res.status(500).json({ error: 'Stress test failed' });
  }
});

app.post('/api/counterfactual', upload.single('image'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No image file' });
  
  // FIX 3: Use Dynamic URL
  const pythonApiUrl = getAiUrl('/api/counterfactual');

  try {
    const formData = new FormData();
    formData.append('image', fs.createReadStream(req.file.path));
    const response = await axios.post(pythonApiUrl, formData, { headers: { ...formData.getHeaders() }, timeout: 600000 });
    fs.unlinkSync(req.file.path);
    res.json(response.data);
  } catch (error) {
    if (fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
    res.status(500).json({ error: 'AI service failed' });
  }
});

// --- FIXED PDF REPORT (WITH COUNTERFACTUALS) ---
app.post('/api/report', (req, res) => {
  const { verdict, confidence, reason, fidelity, timestamp, originalImage, heatmapImage, robustness, counterfactual } = req.body;

  const dataToSign = `${verdict}-${confidence}-${timestamp}-${process.env.JWT_SECRET}`;
  const signature = crypto.createHash('sha256').update(dataToSign).digest('hex');

  const doc = new PDFDocument({ margin: 30, size: 'A4' }); 
  res.setHeader('Content-Type', 'application/pdf');
  res.setHeader('Content-Disposition', `attachment; filename=analysis_report_${Date.now()}.pdf`);
  doc.pipe(res);

  // --- PAGE 1: MAIN ANALYSIS ---
  
  // 1. HEADER
  doc.rect(0, 0, 595, 60).fill('#007bff'); 
  doc.fontSize(18).fill('#ffffff').text('DEEPFAKE DETECTION REPORT', 30, 20);
  doc.fontSize(9).text(`Generated: ${new Date(timestamp).toLocaleString()}`, 30, 45);

  // 2. SUMMARY BOX
  let y = 80; 
  doc.roundedRect(30, y, 535, 70, 5).stroke('#333333');
  
  doc.fill('#000000').fontSize(12).text('Verdict:', 45, y + 15);
  const color = verdict === 'FAKE' ? '#dc3545' : '#28a745';
  doc.fontSize(20).fill(color).text(verdict, 45, y + 30);
  
  doc.fill('#000000').fontSize(10).text(`Confidence: ${confidence}%`, 200, y + 20);
  doc.text(`Fidelity: ${fidelity}`, 200, y + 35);
  
  y += 90;

  // 3. IMAGES (Heatmap)
  doc.fontSize(12).fill('#000000').text('Visual Evidence', 30, y, { underline: true });
  y += 20;

  if (originalImage && heatmapImage) {
    try {
      const cleanBase64 = (str) => str.replace(/^data:image\/[a-z]+;base64,/, "");
      const imgBuffer1 = Buffer.from(cleanBase64(originalImage), 'base64');
      const imgBuffer2 = Buffer.from(cleanBase64(heatmapImage), 'base64');

      doc.image(imgBuffer1, 30, y, { width: 180, height: 180, fit: [180, 180] });
      doc.image(imgBuffer2, 240, y, { width: 180, height: 180, fit: [180, 180] });
      
      doc.fontSize(8).text('Input', 30, y + 185, { width: 180, align: 'center' });
      doc.text('AI Analysis Heatmap', 240, y + 185, { width: 180, align: 'center' });
      
      y += 210; 
    } catch (e) {
      doc.text("(Image Error)", 30, y);
    }
  }

  // 4. FINDINGS
  doc.fontSize(12).text('Findings', 30, y, { underline: true });
  y += 15;
  doc.fontSize(9).text(reason, 30, y, { width: 535, height: 80, align: 'justify' });
  y += 90;

  // 5. ROBUSTNESS
  if (robustness) {
    doc.fontSize(12).text('Stress Test Results', 30, y, { underline: true });
    y += 15;
    doc.fontSize(8).font('Courier');
    
    const tests = Object.entries(robustness).filter(([key]) => key !== 'original');
    tests.forEach(([key, result]) => {
      const label = key.charAt(0).toUpperCase() + key.slice(1);
      doc.text(`[${label.padEnd(12)}] Verdict: ${result.verdict} (${result.confidence}%)`, 30, y);
      y += 12;
    });
    doc.font('Helvetica');
  }

  // --- PAGE 2: COUNTERFACTUALS (New Section) ---
  if (counterfactual) {
    doc.addPage(); // Clean new page
    
    // Header
    doc.rect(0, 0, 595, 40).fill('#6f42c1'); // Purple header for advanced
    doc.fontSize(14).fill('#ffffff').text('ADVANCED ANALYSIS: COUNTERFACTUALS', 30, 15);
    
    let cy = 60;
    doc.fill('#000000').fontSize(12).text('What-If Simulation', 30, cy, { underline: true });
    cy += 20;
    
    doc.fontSize(10).text(counterfactual.difference_text || "Visual modification analysis.", 30, cy);
    cy += 30;

    try {
      const cleanBase64 = (str) => str.replace(/^data:image\/[a-z]+;base64,/, "");
      const cfImg1 = Buffer.from(cleanBase64(counterfactual.original_image), 'base64');
      const cfImg2 = Buffer.from(cleanBase64(counterfactual.counterfactual_image), 'base64');

      doc.image(cfImg1, 30, cy, { width: 200, height: 200, fit: [200, 200] });
      doc.image(cfImg2, 260, cy, { width: 200, height: 200, fit: [200, 200] });
      
      doc.fontSize(9).text('Original State', 30, cy + 210, { width: 200, align: 'center' });
      doc.text('Modified State (Flipped Verdict)', 260, cy + 210, { width: 200, align: 'center' });

    } catch (e) {
      doc.text("(Counterfactual images unavailable)", 30, cy);
    }
  }

  // 6. FOOTER (On the last page)
  const bottom = 780;
  doc.rect(30, bottom, 535, 40).fill('#f8f9fa');
  doc.fill('#333').fontSize(6)
      .text(`HASH: ${signature}`, 35, bottom + 15, { width: 525 });
  doc.text('Automated Report. Not Forensic Proof.', 35, bottom + 30, { align: 'center' });

  doc.end();
});

const server = app.listen(port, () => {
  console.log(`Node.js main backend listening at http://localhost:${port}`);
});
server.setTimeout(600000);