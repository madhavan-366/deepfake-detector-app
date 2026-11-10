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
require('dotenv').config({ path: path.join(__dirname, '../.env') });

// Connect DB (with graceful local fallback if Atlas cluster is paused or unresolved)
const connectDB = async () => {
  const primaryUri = process.env.MONGODB_URI || process.env.MONGO_URI;
  if (primaryUri) {
    try {
      await mongoose.connect(primaryUri, { serverSelectionTimeoutMS: 4000 });
      console.log('MongoDB connected successfully.');
      return;
    } catch (err) {
      console.error('MongoDB connection error:', err.message);
      if (err.message && (err.message.includes('ENOTFOUND') || err.message.includes('querySrv'))) {
        console.warn('⚠️  MongoDB Atlas hostname could not be resolved (querySrv ENOTFOUND).');
        console.warn('   The Atlas cluster may be PAUSED, deleted, or the hostname in environment variables is incorrect.');
      }
    }
  } else {
    console.warn('⚠️  Neither MONGODB_URI nor MONGO_URI is set in environment variables.');
  }
  try {
    const localUri = 'mongodb://127.0.0.1:27017/deepfakeappdb';
    console.log('🔄 Attempting connection to local MongoDB fallback (127.0.0.1:27017)...');
    await mongoose.connect(localUri, { serverSelectionTimeoutMS: 3000 });
    console.log('✅ Connected to local MongoDB fallback successfully (mongodb://127.0.0.1:27017/deepfakeappdb).');
  } catch (localErr) {
    console.error('❌ Could not connect to local MongoDB fallback:', localErr.message);
  }
};
connectDB();

const User = require('./models/User');

const app = express();
const port = process.env.PORT || 3000;
app.use(cors());

// --- CRITICAL: Increase Payload Limit ---
app.use(express.json({ limit: '50mb' })); 
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// --- HEALTH CHECK ENDPOINTS ---
app.get('/', (req, res) => {
  res.json({
    status: 'online',
    service: 'Deepfake Detector Backend Gateway',
    database: mongoose.connection.readyState === 1 ? 'connected' : 'disconnected'
  });
});

app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    database: mongoose.connection.readyState === 1 ? 'connected' : 'disconnected'
  });
});

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
// Secure authentication middleware for protected routes
const authMiddleware = (req, res, next) => {
  let token = req.header('x-auth-token');
  const authHeader = req.header('Authorization');
  if (!token && authHeader && authHeader.startsWith('Bearer ')) {
    token = authHeader.slice(7).trim();
  }
  if (!token) {
    return res.status(401).json({ msg: 'Authentication required. No token provided.' });
  }
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = decoded.user;
    next();
  } catch (err) {
    return res.status(401).json({ msg: 'Session expired or invalid. Please sign in again.' });
  }
};

app.post('/api/auth/register', async (req, res) => {
  const { email, password } = req.body;
  try {
    let user = await User.findOne({ email });
    if (user) return res.status(400).json({ msg: 'User already exists' });
    user = new User({ email, password });
    const salt = await bcrypt.genSalt(10);
    user.password = await bcrypt.hash(password, salt);
    await user.save();
    const payload = { user: { id: user.id, email: user.email } };
    jwt.sign(payload, process.env.JWT_SECRET, { expiresIn: '1h' }, (err, token) => {
      if (err) throw err;
      res.json({ token });
    });
  } catch (err) { console.error('Registration error:', err); res.status(500).send('Server error'); }
});

app.post('/api/auth/login', async (req, res) => {
  const { email, password } = req.body;
  try {
    let user = await User.findOne({ email });
    if (!user) return res.status(400).json({ msg: 'Invalid credentials' });
    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) return res.status(400).json({ msg: 'Invalid credentials' });
    const payload = { user: { id: user.id, email: user.email } };
    jwt.sign(payload, process.env.JWT_SECRET, { expiresIn: '1h' }, (err, token) => {
      if (err) throw err;
      res.json({ token });
    });
  } catch (err) { res.status(500).send('Server error'); }
});

// Authenticated current user endpoint
app.get('/api/auth/me', authMiddleware, async (req, res) => {
  try {
    const user = await User.findById(req.user.id).select('-password');
    if (!user) return res.status(404).json({ msg: 'User not found' });
    res.json({ user });
  } catch (err) {
    console.error('Fetch user error:', err);
    res.status(500).send('Server error');
  }
});

// --- CHANGE PASSWORD ENDPOINT ---
// Requires valid authentication token; re-authenticates current password before updating
app.post('/api/auth/change-password', authMiddleware, async (req, res) => {
  const { currentPassword, newPassword, confirmPassword } = req.body;

  if (!currentPassword || !newPassword) {
    return res.status(400).json({ msg: 'Current password and new password are required.' });
  }

  if (typeof newPassword !== 'string' || newPassword.length < 6) {
    return res.status(400).json({ msg: 'New password must be at least 6 characters long.' });
  }

  if (confirmPassword !== undefined && newPassword !== confirmPassword) {
    return res.status(400).json({ msg: 'New passwords do not match.' });
  }

  if (currentPassword === newPassword) {
    return res.status(400).json({ msg: 'New password cannot be the same as your current password.' });
  }

  try {
    const user = await User.findById(req.user.id);
    if (!user) {
      return res.status(404).json({ msg: 'User account not found.' });
    }

    // Re-authenticate by verifying current password
    const isMatch = await bcrypt.compare(currentPassword, user.password);
    if (!isMatch) {
      return res.status(400).json({ msg: 'The current password you entered is incorrect.' });
    }

    // Hash new password using bcrypt
    const salt = await bcrypt.genSalt(10);
    user.password = await bcrypt.hash(newPassword, salt);
    await user.save();

    return res.json({ msg: 'Password updated successfully.' });
  } catch (err) {
    console.error('Change password error:', err);
    return res.status(500).json({ msg: 'Server error while updating password.' });
  }
});

// --- RATE LIMITING STORE & HELPER FOR PASSWORD RESET ---
const resetRateLimitStore = new Map();

const getClientIp = (req) => {
  return (
    req.headers['x-forwarded-for']?.split(',')[0].trim() ||
    req.socket.remoteAddress ||
    req.ip ||
    '127.0.0.1'
  );
};

const checkResetRateLimit = (key, maxAttempts = 5, windowMs = 15 * 60 * 1000) => {
  const now = Date.now();
  const record = resetRateLimitStore.get(key) || { count: 0, resetTime: now + windowMs };

  // If window expired, reset counter
  if (now > record.resetTime) {
    record.count = 0;
    record.resetTime = now + windowMs;
  }

  record.count += 1;
  resetRateLimitStore.set(key, record);

  const isRateLimited = record.count > maxAttempts;
  const timeRemainingSeconds = Math.max(0, Math.ceil((record.resetTime - now) / 1000));

  return { isRateLimited, timeRemainingSeconds };
};

// Periodic cleanup of expired rate limit keys every 10 minutes
setInterval(() => {
  const now = Date.now();
  for (const [key, val] of resetRateLimitStore.entries()) {
    if (now > val.resetTime) {
      resetRateLimitStore.delete(key);
    }
  }
}, 10 * 60 * 1000).unref();

// --- UNRESTRICTED IN-APP FORGOT/RESET PASSWORD ENDPOINT ---
// Allows verified registered accounts to reset credentials directly without email/SMTP links.
// Incorporates rate-limiting, strict email verification, strong password requirements,
// and bcrypt password hashing.
app.post('/api/auth/reset-password', async (req, res) => {
  const { email, newPassword, confirmPassword } = req.body;

  // 1. Basic validation
  if (!email || typeof email !== 'string') {
    return res.status(400).json({ msg: 'Please enter your registered email address.' });
  }

  const cleanEmail = email.trim().toLowerCase();
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(cleanEmail)) {
    return res.status(400).json({ msg: 'Please enter a valid email address format.' });
  }

  if (!newPassword || typeof newPassword !== 'string') {
    return res.status(400).json({ msg: 'Please enter a new password.' });
  }

  if (newPassword.length < 6) {
    return res.status(400).json({ msg: 'New password must be at least 6 characters long.' });
  }

  if (confirmPassword === undefined || newPassword !== confirmPassword) {
    return res.status(400).json({ msg: 'New passwords do not match. Please ensure both passwords match.' });
  }

  // 2. Rate limiting check (per IP & per Email)
  const clientIp = getClientIp(req);
  const ipLimit = checkResetRateLimit(`ip:${clientIp}`, 10, 15 * 60 * 1000);
  if (ipLimit.isRateLimited) {
    return res.status(429).json({
      msg: `Too many password reset requests from this network. Please wait ${Math.ceil(ipLimit.timeRemainingSeconds / 60)} minute(s) before trying again.`
    });
  }

  const emailLimit = checkResetRateLimit(`email:${cleanEmail}`, 5, 15 * 60 * 1000);
  if (emailLimit.isRateLimited) {
    return res.status(429).json({
      msg: `Too many reset attempts for this email address. Please wait ${Math.ceil(emailLimit.timeRemainingSeconds / 60)} minute(s) before trying again.`
    });
  }

  try {
    // 3. Verify that the email belongs to an existing registered account
    const user = await User.findOne({ email: cleanEmail });
    if (!user) {
      return res.status(404).json({
        msg: 'No registered account found with this email address. Please verify your email or sign up.'
      });
    }

    // 4. Verify that new password is not identical to current password
    const isSamePassword = await bcrypt.compare(newPassword, user.password);
    if (isSamePassword) {
      return res.status(400).json({
        msg: 'New password cannot be the same as your current password. Please choose a different password.'
      });
    }

    // 5. Hash new password securely with bcrypt
    const salt = await bcrypt.genSalt(10);
    user.password = await bcrypt.hash(newPassword, salt);
    await user.save();

    console.log(`[AUTH] Password successfully reset for user: ${cleanEmail}`);
    return res.json({
      msg: 'Password has been securely reset. You can now log in with your new password.'
    });
  } catch (err) {
    console.error('Reset password error:', err);
    return res.status(500).json({ msg: 'Server error while resetting password. Please try again later.' });
  }
});

// --- HELPER FOR URL ---
// Normalizes base URL, trims whitespace, and strips trailing slashes
const getAiUrl = (endpoint) => {
  const rawBase = (process.env.AI_SERVICE_URL || 'http://127.0.0.1:5000').trim();
  const baseUrl = rawBase.replace(/\/+$/, '');
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  return `${baseUrl}${cleanEndpoint}`;
};

// --- AI ROUTES ---
app.post('/api/detect', upload.single('image'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No image file' });
  
  const pythonApiUrl = getAiUrl('/api/detect');

  try {
    const formData = new FormData();
    formData.append('image', fs.createReadStream(req.file.path));
    if (req.body.mode) formData.append('mode', req.body.mode);
    console.log(`Sending request to Python API at: ${pythonApiUrl}`);
    const response = await axios.post(pythonApiUrl, formData, {
      headers: { ...formData.getHeaders() },
      timeout: 90000 // 90s timeout to protect against cold-starts within Render's gateway limits
    });
    fs.unlinkSync(req.file.path);
    res.json(response.data);
  } catch (error) {
    if (fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
    const status = error.response ? error.response.status : (error.code || 'UNKNOWN');
    const errorDetail = error.response?.data?.error || error.message;
    console.error(`AI Error [/api/detect] (${status}):`, errorDetail);
    res.status(500).json({ error: 'AI service failed', detail: errorDetail });
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
    console.error("Chat Error:", error.message || error);
    res.status(500).json({ reply: "Sorry, I'm having trouble connecting to my brain right now." });
  }
});

app.post('/api/stress-test', upload.single('image'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No image file' });
  
  const pythonApiUrl = getAiUrl('/api/stress-test');

  try {
    const formData = new FormData();
    formData.append('image', fs.createReadStream(req.file.path));
    const response = await axios.post(pythonApiUrl, formData, {
      headers: { ...formData.getHeaders() },
      timeout: 90000
    });
    fs.unlinkSync(req.file.path);
    res.json(response.data);
  } catch (error) {
    if (fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
    const status = error.response ? error.response.status : (error.code || 'UNKNOWN');
    const errorDetail = error.response?.data?.error || error.message;
    console.error(`AI Error [/api/stress-test] (${status}):`, errorDetail);
    res.status(500).json({ error: 'Stress test failed', detail: errorDetail });
  }
});

app.post('/api/counterfactual', upload.single('image'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No image file' });
  
  const pythonApiUrl = getAiUrl('/api/counterfactual');
  const timeoutMs = process.env.NODE_ENV === 'production' ? 95000 : 600000;

  try {
    const formData = new FormData();
    formData.append('image', fs.createReadStream(req.file.path));
    const response = await axios.post(pythonApiUrl, formData, {
      headers: { ...formData.getHeaders() },
      timeout: timeoutMs
    });
    fs.unlinkSync(req.file.path);
    res.json(response.data);
  } catch (error) {
    if (fs.existsSync(req.file.path)) fs.unlinkSync(req.file.path);
    const status = error.response ? error.response.status : (error.code || 'UNKNOWN');
    const errorDetail = error.response?.data?.error || error.message;
    console.error(`AI Error [/api/counterfactual] (${status}):`, errorDetail);
    res.status(500).json({ error: 'AI service failed', detail: errorDetail });
  }
});

// --- PROFESSIONAL DIGITAL FORENSIC REPORT GENERATION ---
app.post('/api/report', (req, res) => {
  const {
    verdict = 'REAL',
    confidence = 0,
    reason = '',
    fidelity = 'MEDIUM',
    timestamp = new Date().toISOString(),
    originalImage = null,
    heatmapImage = null,
    robustness = null,
    counterfactual = null,
    metadata = null,
    userRole = 'novice'
  } = req.body;

  const isFake = String(verdict).toUpperCase() === 'FAKE';
  const confidenceNum = typeof confidence === 'number' ? confidence : parseFloat(confidence) || 0;
  const normalizedFidelity = fidelity ? String(fidelity).toUpperCase() : 'MEDIUM';

  // Digital verification signature
  const dataToSign = `${verdict}-${confidenceNum}-${timestamp}-${process.env.JWT_SECRET || 'secret'}`;
  const signature = crypto.createHash('sha256').update(dataToSign).digest('hex');

  // Format date helper
  const formatDate = (isoString) => {
    try {
      const d = new Date(isoString || Date.now());
      return d.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        timeZoneName: 'short'
      });
    } catch {
      return new Date().toISOString();
    }
  };

  const cleanBase64 = (str) => {
    if (!str || typeof str !== 'string') return null;
    return str.replace(/^data:image\/[a-z]+;base64,/, '');
  };

  // PDF Document with buffered pages for dynamic pagination
  const doc = new PDFDocument({
    size: 'A4',
    margin: 36,
    bufferPages: true,
    autoFirstPage: true
  });

  res.setHeader('Content-Type', 'application/pdf');
  res.setHeader('Content-Disposition', `attachment; filename=Deepfake_Forensic_Report_${Date.now()}.pdf`);
  doc.pipe(res);

  // Professional Color Palette (White background, charcoal text, emerald/teal accents, no blue/purple)
  const colors = {
    bgCard: '#F8FAFC',
    bgSubtle: '#F1F5F9',
    border: '#E2E8F0',
    borderDark: '#CBD5E1',
    textPrimary: '#0F172A',
    textSecondary: '#334155',
    textMuted: '#64748B',
    accentTeal: '#0D9488',
    accentEmerald: '#10B981',
    accentEmeraldDark: '#059669',
    emeraldBg: '#ECFDF5',
    emeraldBorder: '#A7F3D0',
    statusFake: '#DC2626',
    fakeBg: '#FEF2F2',
    fakeBorder: '#FECACA',
    warnAmber: '#F59E0B',
    warnDark: '#B45309',
    warnBg: '#FFFBEB',
    warnBorder: '#FDE68A'
  };

  const pageWidth = 595.28;
  const pageHeight = 841.89;
  const margin = 36;
  const contentWidth = pageWidth - margin * 2; // 523.28 pt

  let y = margin;

  // Space management helper
  const ensureSpace = (neededHeight) => {
    if (y + neededHeight > pageHeight - margin - 25) {
      doc.addPage();
      y = margin + 18;
      return true;
    }
    return false;
  };

  // ==========================================
  // 1. REPORT HEADER
  // ==========================================
  doc.rect(margin, y, contentWidth, 3).fill(colors.accentTeal);
  y += 10;

  doc.font('Helvetica-Bold').fontSize(7.5).fillColor(colors.accentTeal)
    .text('DEEPFAKE DETECTOR • DIGITAL FORENSICS LAB', margin, y, { characterSpacing: 0.8 });
  y += 11;

  doc.font('Helvetica-Bold').fontSize(17).fillColor(colors.textPrimary)
    .text('DEEPFAKE DETECTION REPORT', margin, y);
  y += 20;

  doc.font('Helvetica').fontSize(8.5).fillColor(colors.textMuted)
    .text('AI-Powered Image Analysis & Authenticity Verification', margin, y);

  const headerRightY = margin + 10;
  doc.font('Helvetica-Bold').fontSize(8).fillColor(colors.textSecondary)
    .text(`Report ID: #DF-${signature.slice(0, 8).toUpperCase()}`, margin, headerRightY, { align: 'right', width: contentWidth });

  doc.font('Helvetica').fontSize(7.5).fillColor(colors.textMuted)
    .text(`Generated: ${formatDate(timestamp)}`, margin, headerRightY + 12, { align: 'right', width: contentWidth });

  const modeLabel = userRole === 'expert' ? 'Expert Forensic (LIME)' : 'Standard Scan (Grad-CAM)';
  doc.font('Helvetica-Bold').fontSize(7.5).fillColor(colors.accentTeal)
    .text(`Analysis Mode: ${modeLabel}`, margin, headerRightY + 23, { align: 'right', width: contentWidth });

  y += 14;
  doc.strokeColor(colors.border).lineWidth(1).moveTo(margin, y).lineTo(margin + contentWidth, y).stroke();
  y += 12;

  // ==========================================
  // 2. ANALYSIS SUMMARY CARD
  // ==========================================
  const summaryCardH = 64;
  doc.roundedRect(margin, y, contentWidth, summaryCardH, 5).fillAndStroke(colors.bgCard, colors.border);

  const accentColor = isFake ? colors.statusFake : colors.accentEmerald;
  doc.roundedRect(margin, y, 4, summaryCardH, 2).fill(accentColor);

  const col1X = margin + 16;
  doc.font('Helvetica-Bold').fontSize(6.5).fillColor(colors.textMuted)
    .text('DETECTION VERDICT', col1X, y + 11, { characterSpacing: 0.5 });

  const pillW = 165;
  const pillH = 22;
  const pillY = y + 22;
  const pillBg = isFake ? colors.fakeBg : colors.emeraldBg;
  const pillBorder = isFake ? colors.fakeBorder : colors.emeraldBorder;
  const pillTextColor = isFake ? colors.statusFake : colors.accentEmeraldDark;
  const pillText = isFake ? 'MANIPULATION DETECTED' : 'AUTHENTIC MEDIA VERIFIED';

  doc.roundedRect(col1X, pillY, pillW, pillH, 4).fillAndStroke(pillBg, pillBorder);
  doc.font('Helvetica-Bold').fontSize(7.5).fillColor(pillTextColor)
    .text(pillText, col1X, pillY + 6, { width: pillW, align: 'center' });

  doc.font('Helvetica').fontSize(6.5).fillColor(colors.textMuted)
    .text(`Model Classification: ${verdict.toUpperCase()}`, col1X, pillY + 25);

  const col2X = margin + 205;
  doc.font('Helvetica-Bold').fontSize(6.5).fillColor(colors.textMuted)
    .text('CONFIDENCE SCORE', col2X, y + 11, { characterSpacing: 0.5 });

  doc.font('Helvetica-Bold').fontSize(18).fillColor(colors.textPrimary)
    .text(`${confidenceNum.toFixed(1)}%`, col2X, y + 21);

  const meterW = 95;
  const meterH = 3.5;
  const meterY = y + 46;
  doc.roundedRect(col2X, meterY, meterW, meterH, 2).fill(colors.border);
  const filledW = Math.max(2, (meterW * Math.min(Math.max(confidenceNum, 0), 100)) / 100);
  doc.roundedRect(col2X, meterY, filledW, meterH, 2).fill(accentColor);

  const col3X = margin + 345;
  doc.font('Helvetica-Bold').fontSize(6.5).fillColor(colors.textMuted)
    .text('EXPLANATION FIDELITY', col3X, y + 11, { characterSpacing: 0.5 });

  const fidelityColor = normalizedFidelity === 'HIGH' ? colors.accentEmeraldDark : normalizedFidelity === 'LOW' ? colors.statusFake : colors.warnDark;
  doc.font('Helvetica-Bold').fontSize(10).fillColor(fidelityColor)
    .text(normalizedFidelity, col3X, y + 22);

  doc.font('Helvetica').fontSize(7).fillColor(colors.textMuted)
    .text('Local attribution accuracy', col3X, y + 36);

  doc.font('Helvetica').fontSize(7).fillColor(colors.textSecondary)
    .text('Architecture: Dual-Stream Xception', col3X, y + 47);

  y += summaryCardH + 14;

  // ==========================================
  // 3. VISUAL EVIDENCE (SIDE-BY-SIDE)
  // ==========================================
  doc.font('Helvetica-Bold').fontSize(9).fillColor(colors.textPrimary)
    .text('VISUAL EVIDENCE & ATTRIBUTION HEATMAP', margin, y);
  y += 11;

  const imgBoxGap = 15;
  const imgBoxW = (contentWidth - imgBoxGap) / 2;
  const imgBoxH = 140;

  const box1X = margin;
  const box1Y = y;
  doc.roundedRect(box1X, box1Y, imgBoxW, imgBoxH, 4).fillAndStroke(colors.bgSubtle, colors.border);

  let originalLoaded = false;
  if (originalImage) {
    try {
      const cleanB64 = cleanBase64(originalImage);
      if (cleanB64) {
        const buf = Buffer.from(cleanB64, 'base64');
        doc.image(buf, box1X + 4, box1Y + 4, {
          fit: [imgBoxW - 8, imgBoxH - 8],
          align: 'center',
          valign: 'center'
        });
        originalLoaded = true;
      }
    } catch (err) {
      console.warn('Original image render warning:', err.message);
    }
  }
  if (!originalLoaded) {
    doc.font('Helvetica').fontSize(7.5).fillColor(colors.textMuted)
      .text('Input image preview unavailable', box1X, box1Y + imgBoxH / 2 - 5, { width: imgBoxW, align: 'center' });
  }

  const box2X = margin + imgBoxW + imgBoxGap;
  const box2Y = y;
  doc.roundedRect(box2X, box2Y, imgBoxW, imgBoxH, 4).fillAndStroke(colors.bgSubtle, colors.border);

  let heatmapLoaded = false;
  if (heatmapImage) {
    try {
      const cleanB64 = cleanBase64(heatmapImage);
      if (cleanB64) {
        const buf = Buffer.from(cleanB64, 'base64');
        doc.image(buf, box2X + 4, box2Y + 4, {
          fit: [imgBoxW - 8, imgBoxH - 8],
          align: 'center',
          valign: 'center'
        });
        heatmapLoaded = true;
      }
    } catch (err) {
      console.warn('Heatmap image render warning:', err.message);
    }
  }
  if (!heatmapLoaded) {
    doc.font('Helvetica').fontSize(7.5).fillColor(colors.textMuted)
      .text('Heatmap attribution unavailable', box2X, box2Y + imgBoxH / 2 - 5, { width: imgBoxW, align: 'center' });
  }

  y += imgBoxH + 6;
  const caption1Title = 'Figure 1: Input Media (Cropped Face)';
  const caption1Desc = 'Facial bounding box detected and aligned via MTCNN for inference.';
  doc.font('Helvetica-Bold').fontSize(7.5).fillColor(colors.textSecondary)
    .text(caption1Title, box1X, y, { width: imgBoxW });
  doc.font('Helvetica').fontSize(7).fillColor(colors.textMuted)
    .text(caption1Desc, box1X, y + 9, { width: imgBoxW, lineGap: 1 });

  const heatmapTypeName = userRole === 'expert' ? 'LIME Superpixel' : 'Grad-CAM Attention';
  const caption2Title = `Figure 2: AI Analysis Heatmap (${heatmapTypeName})`;
  const caption2Desc = 'Highlighted warm regions indicate features with high influence on the verdict.';
  doc.font('Helvetica-Bold').fontSize(7.5).fillColor(colors.textSecondary)
    .text(caption2Title, box2X, y, { width: imgBoxW });
  doc.font('Helvetica').fontSize(7).fillColor(colors.textMuted)
    .text(caption2Desc, box2X, y + 9, { width: imgBoxW, lineGap: 1 });

  y += 28;

  // ==========================================
  // 4. FORENSIC ANALYSIS FINDINGS
  // ==========================================
  doc.font('Helvetica-Bold').fontSize(9).fillColor(colors.textPrimary)
    .text('FORENSIC ANALYSIS FINDINGS', margin, y);
  y += 11;

  const cleanReason = String(reason || 'No detailed findings narrative provided.').trim();
  const numberedPattern = /(\d+\.\s+[^0-9]+)/g;
  const matches = cleanReason.match(numberedPattern);

  const findingsStartY = y;
  const findingsInnerX = margin + 12;
  const findingsInnerW = contentWidth - 24;

  let textCursorY = findingsStartY + 9;

  if (matches && matches.length > 1) {
    const firstIndex = cleanReason.indexOf(matches[0]);
    const intro = firstIndex > 0 ? cleanReason.slice(0, firstIndex).trim() : null;

    if (intro) {
      textCursorY += doc.heightOfString(intro, { width: findingsInnerW, lineGap: 1.5 }) + 6;
    }

    matches.forEach((item) => {
      const cleaned = item.replace(/^\d+\.\s*/, '').trim();
      textCursorY += doc.heightOfString(cleaned, { width: findingsInnerW - 14, lineGap: 1.5 }) + 4;
    });

    const lastMatch = matches[matches.length - 1];
    const afterIndex = cleanReason.indexOf(lastMatch) + lastMatch.length;
    const outro = afterIndex < cleanReason.length ? cleanReason.slice(afterIndex).trim() : null;
    if (outro) {
      textCursorY += doc.heightOfString(outro, { width: findingsInnerW, lineGap: 1.5 }) + 4;
    }
  } else {
    textCursorY += doc.heightOfString(cleanReason, { width: findingsInnerW, lineGap: 2.5 }) + 8;
  }

  const findingsBoxH = textCursorY - findingsStartY + 4;
  doc.roundedRect(margin, findingsStartY, contentWidth, findingsBoxH, 4).fillAndStroke(colors.bgCard, colors.border);

  // Render text cleanly inside card
  textCursorY = findingsStartY + 9;
  if (matches && matches.length > 1) {
    const firstIndex = cleanReason.indexOf(matches[0]);
    const intro = firstIndex > 0 ? cleanReason.slice(0, firstIndex).trim() : null;

    if (intro) {
      doc.font('Helvetica-Bold').fontSize(8).fillColor(colors.textPrimary)
        .text(intro, findingsInnerX, textCursorY, { width: findingsInnerW, lineGap: 1.5 });
      textCursorY += doc.heightOfString(intro, { width: findingsInnerW, lineGap: 1.5 }) + 6;
    }

    matches.forEach((item, idx) => {
      const cleaned = item.replace(/^\d+\.\s*/, '').trim();
      const bulletPrefix = `${idx + 1}.`;

      doc.font('Helvetica-Bold').fontSize(8).fillColor(colors.accentTeal)
        .text(bulletPrefix, findingsInnerX, textCursorY, { width: 14 });

      doc.font('Helvetica').fontSize(8).fillColor(colors.textSecondary)
        .text(cleaned, findingsInnerX + 14, textCursorY, { width: findingsInnerW - 14, lineGap: 1.5 });

      textCursorY += doc.heightOfString(cleaned, { width: findingsInnerW - 14, lineGap: 1.5 }) + 4;
    });

    const lastMatch = matches[matches.length - 1];
    const afterIndex = cleanReason.indexOf(lastMatch) + lastMatch.length;
    const outro = afterIndex < cleanReason.length ? cleanReason.slice(afterIndex).trim() : null;
    if (outro) {
      doc.font('Helvetica-Oblique').fontSize(7.5).fillColor(colors.textMuted)
        .text(outro, findingsInnerX, textCursorY, { width: findingsInnerW, lineGap: 1.5 });
    }
  } else {
    doc.font('Helvetica').fontSize(8).fillColor(colors.textSecondary)
      .text(cleanReason, findingsInnerX, textCursorY, { width: findingsInnerW, lineGap: 2.5, align: 'justify' });
  }

  y = findingsStartY + findingsBoxH + 12;

  // ==========================================
  // 5. ANALYSIS DETAILS & METADATA TABLE
  // ==========================================
  ensureSpace(85);

  doc.font('Helvetica-Bold').fontSize(9).fillColor(colors.textPrimary)
    .text('ANALYSIS DETAILS & METADATA', margin, y);
  y += 10;

  const metaTableY = y;
  const metaTableW = contentWidth;
  const metaRowH = 16;
  const metaCol1W = metaTableW * 0.5;

  const metaItems = [
    { label: 'Evaluation Date & Time', val: formatDate(timestamp) },
    { label: 'Model Architecture', val: 'Dual-Stream Xception (Consensus TTA)' },
    { label: 'Face Preprocessing', val: 'MTCNN Margin=80 Face Crop (299×299)' },
    { label: 'Attribution Method', val: `${heatmapTypeName}` },
    { label: 'Source File Name', val: (metadata && metadata.name) ? metadata.name : 'Target Media Upload' },
    { label: 'Original Dimensions', val: (metadata && metadata.dimensions) ? metadata.dimensions : 'Auto-detected 299×299 px' }
  ];

  doc.roundedRect(margin, metaTableY, metaTableW, metaRowH * 3, 4).fillAndStroke(colors.bgCard, colors.border);

  metaItems.forEach((item, index) => {
    const col = index % 2;
    const row = Math.floor(index / 2);
    const cellX = margin + (col === 0 ? 10 : metaCol1W + 10);
    const cellY = metaTableY + row * metaRowH + 4;

    doc.font('Helvetica-Bold').fontSize(7).fillColor(colors.textMuted)
      .text(`${item.label}:`, cellX, cellY);

    const valOffset = 100;
    doc.font('Helvetica').fontSize(7).fillColor(colors.textPrimary)
      .text(item.val, cellX + valOffset, cellY, { width: (metaCol1W - valOffset - 14), ellipsis: true });
  });

  doc.strokeColor(colors.border).lineWidth(0.5);
  doc.moveTo(margin, metaTableY + metaRowH).lineTo(margin + metaTableW, metaTableY + metaRowH).stroke();
  doc.moveTo(margin, metaTableY + metaRowH * 2).lineTo(margin + metaTableW, metaTableY + metaRowH * 2).stroke();
  doc.moveTo(margin + metaCol1W, metaTableY).lineTo(margin + metaCol1W, metaTableY + metaRowH * 3).stroke();

  y = metaTableY + metaRowH * 3 + 12;

  // ==========================================
  // 6. ROBUSTNESS STRESS TESTING (IF PRESENT)
  // ==========================================
  if (robustness && typeof robustness === 'object') {
    const tests = Object.entries(robustness).filter(([k]) => k !== 'original');
    if (tests.length > 0) {
      ensureSpace(95);

      doc.font('Helvetica-Bold').fontSize(9).fillColor(colors.textPrimary)
        .text('ROBUSTNESS STRESS TESTING (IMAGE PERTURBATIONS)', margin, y);
      y += 9;

      const tableY = y;
      const rowH = 15;
      const totalRows = tests.length + 1;

      doc.roundedRect(margin, tableY, contentWidth, rowH * (totalRows + 1), 4).fillAndStroke(colors.bgCard, colors.border);

      doc.rect(margin, tableY, contentWidth, rowH).fill(colors.bgSubtle);
      doc.font('Helvetica-Bold').fontSize(6.5).fillColor(colors.textMuted);
      doc.text('PERTURBATION TEST', margin + 10, tableY + 5);
      doc.text('VERDICT', margin + 200, tableY + 5);
      doc.text('CONFIDENCE', margin + 300, tableY + 5);
      doc.text('STATUS', margin + 410, tableY + 5);

      let currRowY = tableY + rowH;

      if (robustness.original) {
        doc.font('Helvetica-Bold').fontSize(7).fillColor(colors.textPrimary)
          .text('Baseline (Original Input)', margin + 10, currRowY + 4);

        const baseV = robustness.original.verdict || verdict;
        const baseVColor = String(baseV).toUpperCase() === 'FAKE' ? colors.statusFake : colors.accentEmeraldDark;
        doc.font('Helvetica-Bold').fontSize(7).fillColor(baseVColor)
          .text(baseV, margin + 200, currRowY + 4);

        doc.font('Helvetica').fontSize(7).fillColor(colors.textPrimary)
          .text(`${Number(robustness.original.confidence || confidenceNum).toFixed(1)}%`, margin + 300, currRowY + 4);

        doc.font('Helvetica').fontSize(7).fillColor(colors.textMuted)
          .text('Reference Baseline', margin + 410, currRowY + 4);

        doc.strokeColor(colors.border).lineWidth(0.5).moveTo(margin, currRowY + rowH).lineTo(margin + contentWidth, currRowY + rowH).stroke();
        currRowY += rowH;
      }

      tests.forEach(([key, result], idx) => {
        const testName =
          key === 'blur'
            ? 'Gaussian Blur (radius = 2.0)'
            : key === 'noise'
            ? 'Sensor Noise (σ = 0.05)'
            : key === 'compression'
            ? 'JPEG Compression (Q = 50)'
            : key;

        const isFlipped = robustness.original && result.verdict !== robustness.original.verdict;
        const rowVColor = String(result.verdict).toUpperCase() === 'FAKE' ? colors.statusFake : colors.accentEmeraldDark;

        doc.font('Helvetica').fontSize(7).fillColor(colors.textSecondary)
          .text(testName, margin + 10, currRowY + 4);

        doc.font('Helvetica-Bold').fontSize(7).fillColor(rowVColor)
          .text(result.verdict, margin + 200, currRowY + 4);

        doc.font('Helvetica').fontSize(7).fillColor(colors.textPrimary)
          .text(`${Number(result.confidence).toFixed(1)}%`, margin + 300, currRowY + 4);

        if (isFlipped) {
          doc.font('Helvetica-Bold').fontSize(7).fillColor(colors.statusFake)
            .text('Prediction Flipped', margin + 410, currRowY + 4);
        } else {
          doc.font('Helvetica-Bold').fontSize(7).fillColor(colors.accentEmeraldDark)
            .text('Stable Consistency', margin + 410, currRowY + 4);
        }

        if (idx < tests.length - 1) {
          doc.strokeColor(colors.border).lineWidth(0.5).moveTo(margin, currRowY + rowH).lineTo(margin + contentWidth, currRowY + rowH).stroke();
        }
        currRowY += rowH;
      });

      y = currRowY + 12;
    }
  }

  // ==========================================
  // 7. COUNTERFACTUAL ANALYSIS (IF PRESENT)
  // ==========================================
  if (counterfactual && counterfactual.counterfactual_image) {
    doc.addPage();
    y = margin + 18;

    doc.font('Helvetica-Bold').fontSize(9).fillColor(colors.textPrimary)
      .text('COUNTERFACTUAL ANALYSIS (WHAT-IF MODIFICATION)', margin, y);
    y += 5;

    doc.font('Helvetica').fontSize(7.5).fillColor(colors.textMuted)
      .text('Identifies minimum synthetic pixel perturbation needed to flip the model classification.', margin, y);
    y += 14;

    const cfBoxW = (contentWidth - imgBoxGap) / 2;
    const cfBoxH = 140;

    const cf1X = margin;
    doc.roundedRect(cf1X, y, cfBoxW, cfBoxH, 4).fillAndStroke(colors.bgSubtle, colors.border);
    try {
      const buf1 = Buffer.from(cleanBase64(counterfactual.original_image), 'base64');
      doc.image(buf1, cf1X + 4, y + 4, { fit: [cfBoxW - 8, cfBoxH - 8], align: 'center', valign: 'center' });
    } catch (e) { /* ignore */ }

    const cf2X = margin + cfBoxW + imgBoxGap;
    doc.roundedRect(cf2X, y, cfBoxW, cfBoxH, 4).fillAndStroke(colors.bgSubtle, colors.border);
    try {
      const buf2 = Buffer.from(cleanBase64(counterfactual.counterfactual_image), 'base64');
      doc.image(buf2, cf2X + 4, y + 4, { fit: [cfBoxW - 8, cfBoxH - 8], align: 'center', valign: 'center' });
    } catch (e) { /* ignore */ }

    y += cfBoxH + 5;
    doc.font('Helvetica-Bold').fontSize(7.5).fillColor(colors.textSecondary)
      .text(`Original State (${counterfactual.original_pred_text || 'Initial'})`, cf1X, y, { width: cfBoxW, align: 'center' });
    doc.font('Helvetica-Bold').fontSize(7.5).fillColor(colors.accentTeal)
      .text(`Modified State (${counterfactual.target_pred_text || 'Inverted'})`, cf2X, y, { width: cfBoxW, align: 'center' });
    y += 20;

    if (counterfactual.difference_text) {
      doc.font('Helvetica-Bold').fontSize(8).fillColor(colors.textPrimary)
        .text('Key Modifications Applied:', margin, y);
      y += 10;

      const diffLines = counterfactual.difference_text.split('\n').filter(l => l.trim().length > 0);
      diffLines.forEach(line => {
        const cleanLine = line.replace(/^[•\s*-]+/, '').trim();
        doc.font('Helvetica-Bold').fontSize(7.5).fillColor(colors.accentTeal).text('›', margin + 4, y);
        doc.font('Helvetica').fontSize(7.5).fillColor(colors.textSecondary).text(cleanLine, margin + 14, y, { width: contentWidth - 20 });
        y += 10;
      });
      y += 12;
    }
  }

  // ==========================================
  // 8. DISCLAIMER & DIGITAL SIGNATURE
  // ==========================================
  ensureSpace(85);

  const disclaimerBoxY = y;
  const disclaimerBoxH = 54;

  doc.roundedRect(margin, disclaimerBoxY, contentWidth, disclaimerBoxH, 4).fillAndStroke(colors.warnBg, colors.warnBorder);
  doc.roundedRect(margin, disclaimerBoxY, 3, disclaimerBoxH, 1.5).fill(colors.warnDark);

  doc.font('Helvetica-Bold').fontSize(7).fillColor(colors.warnDark)
    .text('RESPONSIBLE-USE FORENSIC DISCLAIMER', margin + 10, disclaimerBoxY + 8, { characterSpacing: 0.5 });

  const disclaimerText =
    'This report provides automated artificial intelligence predictions intended strictly for informational and investigative guidance. Model predictions can be affected by digital compression artifacts, sensor noise, or low-light conditions, and should not be treated as definitive legal proof of authenticity or tampering. Crucial determinations should always be corroborated with independent digital forensic evidence and provenance verification.';

  doc.font('Helvetica').fontSize(6.5).fillColor(colors.textSecondary)
    .text(disclaimerText, margin + 10, disclaimerBoxY + 19, { width: contentWidth - 20, lineGap: 1.2 });

  y = disclaimerBoxY + disclaimerBoxH + 8;

  doc.roundedRect(margin, y, contentWidth, 20, 3).fillAndStroke(colors.bgCard, colors.border);
  doc.font('Helvetica-Bold').fontSize(6).fillColor(colors.textMuted)
    .text('SHA-256 VERIFICATION HASH:', margin + 8, y + 6);
  doc.font('Courier').fontSize(6).fillColor(colors.textSecondary)
    .text(signature, margin + 120, y + 6, { width: contentWidth - 130, ellipsis: true });

  // ==========================================
  // 9. DYNAMIC RUNNING HEADERS & FOOTERS (PAGINATION)
  // ==========================================
  const range = doc.bufferedPageRange();
  for (let i = 0; i < range.count; i++) {
    doc.switchToPage(i);

    if (i > 0) {
      doc.strokeColor(colors.border).lineWidth(0.5).moveTo(margin, 26).lineTo(margin + contentWidth, 26).stroke();
      doc.font('Helvetica-Bold').fontSize(6.5).fillColor(colors.accentTeal)
        .text('DEEPFAKE DETECTOR', margin, 16);
      doc.font('Helvetica').fontSize(6.5).fillColor(colors.textMuted)
        .text('• FORENSIC ANALYSIS REPORT', margin + 85, 16);
      doc.font('Courier').fontSize(6).fillColor(colors.textMuted)
        .text(`REF: #DF-${signature.slice(0, 8).toUpperCase()}`, margin, 16, { width: contentWidth, align: 'right' });
    }

    const footerRuleY = pageHeight - margin - 8;
    doc.strokeColor(colors.border).lineWidth(0.5).moveTo(margin, footerRuleY).lineTo(margin + contentWidth, footerRuleY).stroke();

    doc.font('Helvetica').fontSize(6.5).fillColor(colors.textMuted)
      .text('Deepfake Detector Forensic Media Platform • Automated Evaluation', margin, footerRuleY + 5);

    doc.font('Helvetica-Bold').fontSize(6.5).fillColor(colors.textSecondary)
      .text(`Page ${i + 1} of ${range.count}`, margin, footerRuleY + 5, { width: contentWidth, align: 'right' });
  }

  doc.end();
});

const server = app.listen(port, () => {
  console.log(`Node.js main backend listening at http://localhost:${port}`);
});
server.setTimeout(600000);