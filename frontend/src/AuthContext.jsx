// frontend/src/AuthContext.jsx
import React, { createContext, useState, useEffect, useContext } from 'react';
import axios from 'axios';
import { jwtDecode } from 'jwt-decode'; // You need to install this: npm install jwt-decode

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
    // You might want to re-check auth on window focus or periodically for long sessions
    window.addEventListener('focus', checkAuth);
    return () => window.removeEventListener('focus', checkAuth);
  }, []);

  const login = async (email, password) => {
    try {
      const res = await axios.post('http://localhost:3000/api/auth/login', { email, password });
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
      const res = await axios.post('http://localhost:3000/api/auth/register', { email, password });
      // For register, we just want to succeed, not automatically log in
      // localStorage.setItem('token', res.data.token); // Removed auto-login after register
      // const decoded = jwtDecode(res.data.token);
      // setIsAuthenticated(true);
      // setUser(decoded.user);
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
    // Optionally redirect to login page here if not handled by router
    // window.location.href = '/'; // Simple full page reload to login
  };

  if (loading) {
    // You can render a simple loading spinner here if you wish
    return <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', fontSize: '2em' }}>Loading...</div>;
  }

  return (
    <AuthContext.Provider value={{ isAuthenticated, user, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);