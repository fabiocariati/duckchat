import React, { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Chat from './components/Chat';
import './App.css';

function App() {
  const [state, setState] = useState({
    editIndex: null,
    editText: "",
    messages: [],
    tables: {},
    provider: null,
    model: null,
    apiKey: null
  });

  return (
    <div className="app">
      <Sidebar
        state={state}
        setState={setState}
      />
      <Chat
        state={state}
        setState={setState}
      />
    </div>
  );
}

export default App; 