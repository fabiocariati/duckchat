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

  // When rerunPrompt changes, execute the query
  useEffect(() => {
    if (state.rerunPrompt) {
      // Execute the query with the edited prompt
      const executeEditedPrompt = async () => {
        setLoading(true);
        try {
          const result = await api.executeQuery(
            state.rerunPrompt,
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
                content: 'Result of the edited prompt:',
                sql: result.sql,
                dataframe: result.dataframe
              }
            ],
            rerunPrompt: null
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
            ],
            rerunPrompt: null
          }));
        } finally {
          setLoading(false);
        }
      };
      
      executeEditedPrompt();
    }
  }, [state.rerunPrompt, state.provider, state.model, state.apiKey]);

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
      // Send the SQL directly as the query
      const result = await api.executeQuery(
        editedSql,
        state.provider,
        state.model,
        state.apiKey
      );
      
      // Update the current message with new results instead of replacing all messages
      const updatedMessages = [...state.messages];
      updatedMessages[index] = {
        role: 'assistant',
        content: 'Result of the edited SQL:',
        sql: editedSql,  // Use the edited SQL directly
        dataframe: result.dataframe
      };
      
      setState({
        ...state,
        messages: updatedMessages,
        editIndex: null,
        editText: ""
      });
      
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
    // Create a copy of all messages
    const updatedMessages = [...state.messages];
    
    // Update the edited message content
    updatedMessages[index] = { 
      ...updatedMessages[index], 
      content: newContent 
    };
    
    // If this is a user message that has a response, we need to regenerate the response
    const isUserMessage = updatedMessages[index].role === 'user';
    
    setState({
      ...state,
      messages: updatedMessages,
      editIndex: null,
      editText: "",
      // Set rerunPrompt if this is a user message that needs a new response
      rerunPrompt: isUserMessage ? newContent : null
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