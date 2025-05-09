import React, { useState, useEffect } from 'react';
import api from '../services/api';
import './Sidebar.css';

const Sidebar = ({ state, setState }) => {
  const [providers, setProviders] = useState([]);
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Fetch available providers when component mounts
    const fetchProviders = async () => {
      try {
        const providersList = await api.getProviders();
        setProviders(providersList);
        if (providersList.length > 0) {
          handleProviderChange(providersList[0]);
        }
      } catch (error) {
        console.error("Error fetching providers:", error);
      }
    };

    fetchProviders();
  }, []);

  const handleProviderChange = async (provider) => {
    try {
      setState({ ...state, provider });
      const modelsList = await api.getModels(provider);
      setModels(modelsList);
      if (modelsList.length > 0) {
        setState(prevState => ({ 
          ...prevState, 
          provider, 
          model: modelsList[0] 
        }));
      }
    } catch (error) {
      console.error("Error fetching models:", error);
    }
  };

  const handleFileUpload = async (event) => {
    const files = event.target.files;
    if (!files.length) return;

    setLoading(true);
    
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      
      if (!state.tables[file.name]) {
        try {
          const result = await api.uploadFile(file);
          setState(prevState => ({
            ...prevState,
            tables: {
              ...prevState.tables,
              [file.name]: {
                tableName: result.table_name,
                columns: result.columns
              }
            }
          }));
        } catch (error) {
          console.error(`Error uploading ${file.name}:`, error);
        }
      }
    }
    
    setLoading(false);
    // Reset input to allow uploading the same file again
    event.target.value = null;
  };

  return (
    <div className="sidebar">
      <div className="sidebar-section">
        <h2>⚙️ Settings</h2>
        
        <label>Provider</label>
        <select 
          value={state.provider || ''}
          onChange={(e) => handleProviderChange(e.target.value)}
        >
          {providers.map(provider => (
            <option key={provider} value={provider}>{provider}</option>
          ))}
        </select>
        
        <label>Model</label>
        <select 
          value={state.model || ''}
          onChange={(e) => setState({ ...state, model: e.target.value })}
        >
          {models.map(model => (
            <option key={model} value={model}>{model}</option>
          ))}
        </select>
        
        {state.provider === 'openai' && (
          <>
            <label>API Key</label>
            <input 
              type="password" 
              value={state.apiKey || ''}
              onChange={(e) => setState({ ...state, apiKey: e.target.value })}
              placeholder="Enter OpenAI API Key"
            />
          </>
        )}
        
        {state.provider === 'ollama' && models.length === 0 && (
          <div className="error">
            No models found. Please add a model: `docker exec -it ollama ollama pull llama3:8b`
          </div>
        )}
      </div>

      <div className="sidebar-section">
        <h2>📂 Files</h2>
        <div className="file-upload">
          <label className="upload-btn">
            Upload CSV/Parquet Files
            <input
              type="file"
              accept=".csv,.parquet"
              multiple
              onChange={handleFileUpload}
              style={{ display: 'none' }}
            />
          </label>
          {loading && <div className="loading">Uploading...</div>}
        </div>
      </div>

      <div className="sidebar-section">
        <h2>📊 Tables</h2>
        {Object.entries(state.tables).map(([filename, details]) => (
          <details key={filename} className="table-details">
            <summary>{filename}</summary>
            <ul>
              {details.columns.map((column, index) => (
                <li key={index}>{column}</li>
              ))}
            </ul>
          </details>
        ))}
      </div>
    </div>
  );
};

export default Sidebar; 