// frontend/src/LoginPage.jsx
import React, { useState } from 'react';
import { useAuth } from './AuthContext'; // Import useAuth
import './LoginPage.css';

function LoginPage() {
  const { login, register } = useAuth(); // Use the login/register functions from context
  
  const [isLoginView, setIsLoginView] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [successMessage, setSuccessMessage] = useState(''); // New state for success message

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccessMessage(''); // Clear previous messages
    
    if (isLoginView) {
      const success = await login(email, password);
      if (!success) {
        setError('Login failed. Please check your email and password.');
      } else {
        // On successful login, AuthContext handles isAuthenticated true
        // and main.jsx will redirect to App component automatically.
      }
    } else { // Sign Up
      const success = await register(email, password);
      if (success) {
        setSuccessMessage('Registration successful! Please sign in.');
        setIsLoginView(true); // Switch to login view after successful registration
        setEmail(''); // Clear form
        setPassword('');
      } else {
        setError('Registration failed. This email might already be registered.');
      }
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-box">
        <h2>Deepfake Detector</h2>
        <p className="auth-subtitle">Sign in or create an account to continue</p>
        
        <div className="auth-toggle">
          <button 
            className={`toggle-btn ${isLoginView ? 'active' : ''}`}
            onClick={() => setIsLoginView(true)}
          >
            Sign In
          </button>
          <button 
            className={`toggle-btn ${!isLoginView ? 'active' : ''}`}
            onClick={() => setIsLoginView(false)}
          >
            Sign Up
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="input-group">
            <label htmlFor="email">Email</label>
            <input 
              type="email" 
              id="email" 
              placeholder="you@example.com" 
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required 
            />
          </div>
          <div className="input-group">
            <label htmlFor="password">Password</label>
            <input 
              type="password" 
              id="password" 
              placeholder="••••••••" 
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required 
            />
          </div>
          {error && <p className="auth-error">{error}</p>}
          {successMessage && <p className="auth-success">{successMessage}</p>} {/* Display success */}
          <button type="submit" className="submit-btn">
            {isLoginView ? 'Sign In' : 'Sign Up'}
          </button>
        </form>
      </div>
    </div>
  );
}

export default LoginPage;