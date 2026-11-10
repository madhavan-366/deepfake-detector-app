// frontend/src/config.js
// Production and Development API base URL configuration

const PRODUCTION_BACKEND_URL = 'https://deepfake-backend-kd5b.onrender.com';
const DEVELOPMENT_BACKEND_URL = 'http://localhost:3000';

const envUrl = import.meta.env.VITE_API_URL;

export const API_BASE_URL = envUrl
  ? envUrl.replace(/\/+$/, '')
  : (import.meta.env.PROD ? PRODUCTION_BACKEND_URL : DEVELOPMENT_BACKEND_URL);
