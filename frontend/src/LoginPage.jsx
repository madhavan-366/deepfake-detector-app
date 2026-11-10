// frontend/src/LoginPage.jsx
import React, { useState } from 'react';
import { useAuth } from './AuthContext';
import { ShieldIcon, MailIcon, LockIcon, AlertTriangleIcon, CheckCircleIcon, SpinnerIcon, KeyIcon } from './components/Icons';
import './LoginPage.css';

function LoginPage() {
  const { login, register, resetPassword } = useAuth();

  // View state: 'signin' | 'signup' | 'recovery'
  const [viewMode, setViewMode] = useState('signin');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  // Password reset fields
  const [resetEmail, setResetEmail] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');

  const [error, setError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  const clearMessages = () => {
    setError('');
    setSuccessMessage('');
  };

  const handleAuthSubmit = async (e) => {
    e.preventDefault();
    clearMessages();

    const cleanEmail = email.trim();
    if (!cleanEmail) {
      setError('Please enter your email address.');
      return;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(cleanEmail)) {
      setError('Please enter a valid email address format (e.g., name@example.com).');
      return;
    }

    if (!password) {
      setError('Please enter your password.');
      return;
    }

    setIsSubmitting(true);

    try {
      if (viewMode === 'signin') {
        const success = await login(cleanEmail, password);
        if (!success) {
          setError('Authentication failed. Please verify your email and password.');
        }
      } else if (viewMode === 'signup') {
        if (password.length < 6) {
          setError('Password must be at least 6 characters long.');
          setIsSubmitting(false);
          return;
        }

        const success = await register(cleanEmail, password);
        if (success) {
          setSuccessMessage('Registration successful! Please sign in with your credentials.');
          setViewMode('signin');
          setPassword('');
        } else {
          setError('Registration failed. This email may already be registered in the system.');
        }
      }
    } catch {
      setError('An unexpected network error occurred. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleResetPasswordSubmit = async (e) => {
    e.preventDefault();
    clearMessages();

    const cleanEmail = resetEmail.trim();
    if (!cleanEmail) {
      setError('Please enter your registered email address.');
      return;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(cleanEmail)) {
      setError('Please enter a valid email address format (e.g., name@example.com).');
      return;
    }

    if (!newPassword) {
      setError('Please enter your new password.');
      return;
    }

    if (newPassword.length < 6) {
      setError('New password must be at least 6 characters long.');
      return;
    }

    if (newPassword !== confirmPassword) {
      setError('Passwords do not match. Please ensure both passwords match.');
      return;
    }

    setIsSubmitting(true);

    try {
      const res = await resetPassword(cleanEmail, newPassword, confirmPassword);
      if (res.success) {
        setSuccessMessage('Password reset successfully! You can now log in with your new password.');
        setEmail(cleanEmail);
        setPassword('');
        setResetEmail('');
        setNewPassword('');
        setConfirmPassword('');
        setViewMode('signin');
      } else {
        setError(res.message || 'Failed to reset password. Please check your credentials.');
      }
    } catch {
      setError('An unexpected network error occurred. Please try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-box">
        {/* Brand Header */}
        <div className="auth-header">
          <div className="auth-logo-badge">
            <ShieldIcon size={26} className="auth-logo-icon" />
          </div>
          <h1 className="auth-title">Deepfake Detector</h1>
          <p className="auth-subtitle">AI-Powered Deepfake Detection &amp; Media Forensics</p>
        </div>

        {/* View Toggle (Only shown on Sign In / Sign Up) */}
        {(viewMode === 'signin' || viewMode === 'signup') && (
          <div className="auth-toggle">
            <button
              type="button"
              className={`toggle-btn ${viewMode === 'signin' ? 'active' : ''}`}
              onClick={() => {
                setViewMode('signin');
                clearMessages();
              }}
            >
              Sign In
            </button>
            <button
              type="button"
              className={`toggle-btn ${viewMode === 'signup' ? 'active' : ''}`}
              onClick={() => {
                setViewMode('signup');
                clearMessages();
              }}
            >
              Sign Up
            </button>
          </div>
        )}

        {/* Status Alerts */}
        {error && (
          <div className="auth-alert error">
            <AlertTriangleIcon size={16} />
            <span>{error}</span>
          </div>
        )}
        {successMessage && (
          <div className="auth-alert success">
            <CheckCircleIcon size={16} />
            <span>{successMessage}</span>
          </div>
        )}

        {/* --- VIEW 1: SIGN IN / SIGN UP FORM --- */}
        {(viewMode === 'signin' || viewMode === 'signup') && (
          <form onSubmit={handleAuthSubmit} className="auth-form">
            <div className="input-group">
              <label htmlFor="auth-email">Email Address</label>
              <div className="input-wrapper">
                <span className="input-icon">
                  <MailIcon size={16} />
                </span>
                <input
                  type="email"
                  id="auth-email"
                  placeholder="name@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  autoComplete="email"
                  required
                  disabled={isSubmitting}
                />
              </div>
            </div>

            <div className="input-group">
              <div className="input-label-row">
                <label htmlFor="auth-password">Password</label>
                {viewMode === 'signin' && (
                  <button
                    type="button"
                    className="forgot-password-link"
                    onClick={() => {
                      setResetEmail(email.trim());
                      setViewMode('recovery');
                      clearMessages();
                    }}
                  >
                    Forgot Password?
                  </button>
                )}
              </div>
              <div className="input-wrapper">
                <span className="input-icon">
                  <LockIcon size={16} />
                </span>
                <input
                  type="password"
                  id="auth-password"
                  placeholder="••••••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  autoComplete={viewMode === 'signin' ? 'current-password' : 'new-password'}
                  required
                  disabled={isSubmitting}
                />
              </div>
            </div>

            <button type="submit" className="submit-btn" disabled={isSubmitting}>
              {isSubmitting ? (
                <>
                  <SpinnerIcon size={16} />
                  <span>{viewMode === 'signin' ? 'Signing In...' : 'Creating Account...'}</span>
                </>
              ) : (
                <span>{viewMode === 'signin' ? 'Sign In' : 'Create Account'}</span>
              )}
            </button>
          </form>
        )}

        {/* --- VIEW 2: PASSWORD RESET FORM --- */}
        {viewMode === 'recovery' && (
          <form onSubmit={handleResetPasswordSubmit} className="auth-form">
            <div className="form-subheading">
              <div className="subheading-icon-wrap">
                <KeyIcon size={20} />
              </div>
              <h3 className="subheading-title">Reset Password</h3>
              <p className="subheading-desc">
                Enter your registered email address and create a new password to regain access to your account.
              </p>
            </div>

            <div className="input-group">
              <label htmlFor="recovery-email">Registered Email Address</label>
              <div className="input-wrapper">
                <span className="input-icon">
                  <MailIcon size={16} />
                </span>
                <input
                  type="email"
                  id="recovery-email"
                  placeholder="name@example.com"
                  value={resetEmail}
                  onChange={(e) => setResetEmail(e.target.value)}
                  autoComplete="email"
                  required
                  disabled={isSubmitting}
                />
              </div>
            </div>

            <div className="input-group">
              <label htmlFor="recovery-new-password">New Password</label>
              <div className="input-wrapper">
                <span className="input-icon">
                  <LockIcon size={16} />
                </span>
                <input
                  type="password"
                  id="recovery-new-password"
                  placeholder="Minimum 6 characters"
                  value={newPassword}
                  onChange={(e) => setNewPassword(e.target.value)}
                  autoComplete="new-password"
                  required
                  disabled={isSubmitting}
                />
              </div>
            </div>

            <div className="input-group">
              <label htmlFor="recovery-confirm-password">Confirm Password</label>
              <div className="input-wrapper">
                <span className="input-icon">
                  <LockIcon size={16} />
                </span>
                <input
                  type="password"
                  id="recovery-confirm-password"
                  placeholder="Re-enter new password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  autoComplete="new-password"
                  required
                  disabled={isSubmitting}
                />
              </div>
            </div>

            <button type="submit" className="submit-btn" disabled={isSubmitting}>
              {isSubmitting ? (
                <>
                  <SpinnerIcon size={16} />
                  <span>Resetting Password...</span>
                </>
              ) : (
                <span>Reset Password</span>
              )}
            </button>

            <button
              type="button"
              className="auth-back-btn"
              disabled={isSubmitting}
              onClick={() => {
                setViewMode('signin');
                clearMessages();
              }}
            >
              Return to Login
            </button>
          </form>
        )}

        {/* Footnote */}
        <div className="auth-footer">
          <span>Deepfake Detector • Media Integrity Platform</span>
        </div>
      </div>
    </div>
  );
}

export default LoginPage;
