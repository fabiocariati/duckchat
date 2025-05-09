import axios from 'axios';

// For development, use absolute URL with the API port
const API_BASE_URL = "http://localhost:8000";

const api = {
  // Get available providers
  getProviders: async () => {
    const response = await axios.get(`${API_BASE_URL}/providers`);
    return response.data.providers;
  },

  // Get available models for selected provider
  getModels: async (provider) => {
    const response = await axios.get(`${API_BASE_URL}/models/${provider}`);
    return response.data.models;
  },

  // Execute query
  executeQuery: async (prompt, provider, model, apiKey = null) => {
    const response = await axios.post(`${API_BASE_URL}/query`, {
      prompt,
      provider,
      model,
      api_key: apiKey
    });
    return response.data;
  },

  // Upload file
  uploadFile: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await axios.post(`${API_BASE_URL}/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return response.data;
  },

  // Get tables
  getTables: async () => {
    const response = await axios.get(`${API_BASE_URL}/tables`);
    return response.data.tables;
  }
};

export default api; 