// frontend/src/main.jsx
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import LoginPage from './LoginPage.jsx';
import { AuthProvider, useAuth } from './AuthContext'; // Import AuthProvider and useAuth
import './index.css';

// A Wrapper component to use AuthContext
const AppWrapper = () => {
  const { isAuthenticated, loading } = useAuth();

  if (loading) {
    // You can render a global loading spinner here
    return <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', fontSize: '2em' }}>Loading Application...</div>;
  }

  // If authenticated, show the main app. Otherwise, show the login page.
  return isAuthenticated ? <App /> : <LoginPage />;
};

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <AuthProvider>
      <AppWrapper />
    </AuthProvider>
  </React.StrictMode>,
);