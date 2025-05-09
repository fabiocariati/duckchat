import React, { useState, useRef, useEffect } from 'react';
import AceEditor from 'react-ace';
import api from '../services/api';
import MessageItem from './MessageItem';
import './Chat.css';

// Import required Ace editor modes and themes
import 'ace-builds/src-noconflict/mode-sql';
import 'ace-builds/src-noconflict/theme-xcode';

const Chat = ({ state, setState }) => {
  const [prompt, setPrompt] = useState('');
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  // Scroll to the bottom when messages change
  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [state.messages]);

  const closeEditChat = () => {
    setState({
      ...state,
      editIndex: null,
      editText: ""
    });
  };

  const handleSendMessage = async (event) => {
    event.preventDefault();
    
    if (!prompt.trim()) return;
    if (Object.keys(state.tables).length === 0) {
      alert("Please upload at least one file first.");
      return;
    }

    // Add user message
    setState({
      ...state,
      messages: [
        ...state.messages,
        { role: 'user', content: prompt }
      ]
    });

    setLoading(true);
    
    try {
      const result = await api.executeQuery(
        prompt,
        state.provider,
        state.model,
        state.apiKey
      );
      
      // Add assistant response
      setState(prevState => ({
        ...prevState,
        messages: [
          ...prevState.messages,
          {
            role: 'assistant',
            content: 'Query and the result:',
            sql: result.sql,
            dataframe: result.dataframe
          }
        ]
      }));
    } catch (error) {
      console.error("Error executing query:", error);
      setState(prevState => ({
        ...prevState,
        messages: [
          ...prevState.messages,
          {
            role: 'assistant',
            content: `Error: ${error.message || 'Failed to execute query'}`,
            type: 'error'
          }
        ]
      }));
    } finally {
      setLoading(false);
      setPrompt('');
    }
  };

  const handleEditSql = async (index, editedSql) => {
    try {
      setLoading(true);
      const result = await api.executeQuery(
        editedSql,
        state.provider,
        state.model,
        state.apiKey
      );
      
      // Replace messages up to this point and add new result
      setState({
        ...state,
        messages: [
          ...state.messages.slice(0, index),
          {
            role: 'assistant',
            content: 'Result of the edited SQL:',
            sql: result.sql,
            dataframe: result.dataframe
          }
        ]
      });
      
      closeEditChat();
    } catch (error) {
      console.error("Error executing edited SQL:", error);
      alert(`Error executing SQL: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleEditMessage = (index) => {
    setState({
      ...state,
      editIndex: index,
      editText: state.messages[index].content
    });
  };

  const handleConfirmEdit = (index, newContent) => {
    // For user messages, we'll replace the content and regenerate the response
    const messages = [...state.messages.slice(0, index)];
    messages[index - 1] = { ...messages[index - 1], content: newContent };
    
    setState({
      ...state,
      messages,
      editIndex: null,
      editText: "",
      rerunPrompt: newContent
    });
  };

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {state.messages.map((msg, i) => (
          <MessageItem
            key={i}
            index={i}
            message={msg}
            state={state}
            onEditMessage={handleEditMessage}
            onConfirmEdit={handleConfirmEdit}
            onEditSql={handleEditSql}
            closeEditChat={closeEditChat}
          />
        ))}
        {loading && (
          <div className="loading-message">
            <div className="spinner"></div>
            <p>Generating SQL...</p>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>
      
      <form className="chat-input" onSubmit={handleSendMessage}>
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter your data query"
          disabled={loading}
        />
        <button type="submit" disabled={loading || !prompt.trim()}>
          Send
        </button>
      </form>
    </div>
  );
};

export default Chat; 