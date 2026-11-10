// frontend/src/main.jsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import LoginPage from './LoginPage.jsx';
import { AuthProvider, useAuth } from './AuthContext';
import './index.css';

const AppWrapper = () => {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    return (
      <div style={{ 
        display: 'flex', 
        flexDirection: 'column', 
        gap: '16px', 
        justifyContent: 'center', 
        alignItems: 'center', 
        height: '100vh', 
        backgroundColor: '#0B0F17', 
        color: '#94A3B8', 
        fontFamily: 'Inter, sans-serif' 
      }}>
        <div style={{ 
          width: '36px', 
          height: '36px', 
          border: '3px solid #243042', 
          borderTopColor: '#10B981', 
          borderRadius: '50%', 
          animation: 'spin 1s linear infinite' 
        }} />
        <span style={{ 
          fontSize: '0.9rem', 
          letterSpacing: '0.08em', 
          textTransform: 'uppercase', 
          color: '#64748B' 
        }}>
          Initializing Deepfake Detector...
        </span>
      </div>
    );
  }

  return isAuthenticated ? <App /> : <LoginPage />;
};

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <AuthProvider>
      <AppWrapper />
    </AuthProvider>
  </React.StrictMode>,
);
