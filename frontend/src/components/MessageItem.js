import React, { useState } from 'react';
import AceEditor from 'react-ace';
import './MessageItem.css';

// Import required Ace editor modes and themes
import 'ace-builds/src-noconflict/mode-sql';
import 'ace-builds/src-noconflict/theme-xcode';

const MessageItem = ({
  index,
  message,
  state,
  onEditMessage,
  onConfirmEdit,
  onEditSql,
  closeEditChat
}) => {
  const [editedText, setEditedText] = useState('');
  const [editedSql, setEditedSql] = useState('');
  
  const isBeingEdited = state.editIndex === index;

  if (message.role === 'user') {
    if (isBeingEdited) {
      return (
        <div className="message user-message edit-mode">
          <input
            type="text"
            defaultValue={message.content}
            onChange={(e) => setEditedText(e.target.value)}
            autoFocus
          />
          <div className="edit-actions">
            <button onClick={() => onConfirmEdit(index, editedText || message.content)}>
              Confirm
            </button>
            <button className="secondary" onClick={closeEditChat}>
              Cancel
            </button>
          </div>
        </div>
      );
    }
    
    return (
      <div className="message user-message">
        <div className="message-content">
          <p>{message.content}</p>
          {state.editIndex === null && (
            <button className="edit" onClick={() => onEditMessage(index)}>
              ✏️ Edit
            </button>
          )}
        </div>
      </div>
    );
  } else if (message.role === 'assistant') {
    return (
      <div className="message assistant-message">
        <div className="message-content">
          {message.type === 'error' ? (
            <div className="error">{message.content}</div>
          ) : (
            <p>{message.content}</p>
          )}
          
          {message.sql && (
            <>
              {!isBeingEdited ? (
                <div className="sql-container">
                  <pre className="sql-code">{message.sql}</pre>
                  {state.editIndex === null && (
                    <button className="edit" onClick={() => {
                      onEditMessage(index);
                      setEditedSql(message.sql);
                    }}>
                      🛠️ Edit SQL
                    </button>
                  )}
                </div>
              ) : (
                <div className="sql-editor">
                  <AceEditor
                    mode="sql"
                    theme="xcode"
                    name={`sql-editor-${index}`}
                    value={editedSql || message.sql}
                    onChange={setEditedSql}
                    fontSize={14}
                    width="100%"
                    height="200px"
                    showPrintMargin={false}
                    showGutter={true}
                    highlightActiveLine={true}
                    setOptions={{
                      enableBasicAutocompletion: true,
                      enableLiveAutocompletion: true,
                      enableSnippets: false,
                      showLineNumbers: true,
                      tabSize: 2,
                    }}
                  />
                  <div className="edit-actions">
                    <button onClick={() => onEditSql(index, editedSql || message.sql)}>
                      Confirm SQL edit
                    </button>
                    <button className="secondary" onClick={closeEditChat}>
                      Cancel edit
                    </button>
                  </div>
                </div>
              )}
            </>
          )}
          
          {message.dataframe && (
            <div className="dataframe-container">
              <table className="dataframe">
                <thead>
                  <tr>
                    {Object.keys(message.dataframe[0] || {}).map((key, i) => (
                      <th key={i}>{key}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {message.dataframe.map((row, rowIndex) => (
                    <tr key={rowIndex}>
                      {Object.values(row).map((cell, cellIndex) => (
                        <td key={cellIndex}>
                          {cell !== null ? String(cell) : 'null'}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    );
  }
  
  return null;
};

export default MessageItem; 