import axios from 'axios';

// 1. Create the Axios instance
const api = axios.create({
  // Get the base URL from the .env file, fallback to localhost if not found
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
});

// 2. Request Interceptor
// This runs before every request is sent
api.interceptors.request.use(
  (config) => {
    // Retrieve the access token from LocalStorage
    const token = localStorage.getItem('access_token');

    // If a token exists, append it to the Authorization header
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    return config;
  },
  (error) => {
    // Handle request errors
    return Promise.reject(error);
  }
);

// 3. Response Interceptor (Optional)
// This runs when a response is received from the backend
api.interceptors.response.use(
  (response) => response,
  (error) => {
    // If the backend returns 401 (Unauthorized), the token might be expired
    if (error.response && error.response.status === 401) {
      console.warn("Session might have expired or token is invalid.");
      // Optional: Redirect to login page or clear local storage here
      // localStorage.removeItem('access_token');
      // window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default api;