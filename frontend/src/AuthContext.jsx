// frontend/src/AuthContext.jsx
import React, { createContext, useState, useEffect, useContext } from 'react';
import axios from 'axios';
import { jwtDecode } from 'jwt-decode';
import { API_BASE_URL } from './config';

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState(null); // Stores user info from token (e.g., user.id)
  const [loading, setLoading] = useState(true); // To prevent showing login page briefly

  useEffect(() => {
    const checkAuth = () => {
      const token = localStorage.getItem('token');
      if (token) {
        try {
          const decoded = jwtDecode(token);
          // Check if token is expired
          if (decoded.exp * 1000 > Date.now()) {
            setIsAuthenticated(true);
            setUser(decoded.user); // Store user ID or other info from token
          } else {
            localStorage.removeItem('token'); // Token expired
            setIsAuthenticated(false);
            setUser(null);
          }
        } catch (error) {
          console.error("Invalid token:", error);
          localStorage.removeItem('token');
          setIsAuthenticated(false);
          setUser(null);
        }
      }
      setLoading(false);
    };

    checkAuth();
    window.addEventListener('focus', checkAuth);
    return () => window.removeEventListener('focus', checkAuth);
  }, []);

  const login = async (email, password) => {
    try {
      const res = await axios.post(`${API_BASE_URL}/api/auth/login`, { email, password });
      localStorage.setItem('token', res.data.token);
      const decoded = jwtDecode(res.data.token);
      setIsAuthenticated(true);
      setUser(decoded.user);
      return true; // Success
    } catch (err) {
      console.error("Login error:", err.response?.data?.msg || err.message);
      return false; // Failure
    }
  };

  const register = async (email, password) => {
    try {
      await axios.post(`${API_BASE_URL}/api/auth/register`, { email, password });
      return true; // Success
    } catch (err) {
      console.error("Registration error:", err.response?.data?.msg || err.message);
      return false; // Failure
    }
  };

  const logout = () => {
    localStorage.removeItem('token');
    setIsAuthenticated(false);
    setUser(null);
  };

  const changePassword = async (currentPassword, newPassword, confirmPassword) => {
    try {
      const token = localStorage.getItem('token');
      if (!token) {
        return { success: false, message: 'You must be signed in to change your password.' };
      }
      const res = await axios.post(
        `${API_BASE_URL}/api/auth/change-password`,
        { currentPassword, newPassword, confirmPassword },
        {
          headers: {
            Authorization: `Bearer ${token}`
          }
        }
      );
      return { success: true, message: res.data?.msg || 'Password updated successfully.' };
    } catch (err) {
      console.error("Change password error:", err.response?.data?.msg || err.message);
      return {
        success: false,
        message: err.response?.data?.msg || 'Failed to update password. Please check your credentials.'
      };
    }
  };

  const resetPassword = async (email, newPassword, confirmPassword) => {
    try {
      const res = await axios.post(`${API_BASE_URL}/api/auth/reset-password`, {
        email,
        newPassword,
        confirmPassword,
      });
      return { success: true, message: res.data?.msg || 'Password reset successfully.' };
    } catch (err) {
      console.error("Reset password error:", err.response?.data?.msg || err.message);
      return {
        success: false,
        message: err.response?.data?.msg || 'Failed to reset password. Please try again.',
      };
    }
  };

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
          Verifying Session...
        </span>
      </div>
    );
  }

  return (
    <AuthContext.Provider value={{ isAuthenticated, user, login, register, logout, changePassword, resetPassword }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
