import React, { useState } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { sql } from '@codemirror/lang-sql';
import { githubLight } from '@uiw/codemirror-theme-github';
import { DataGrid } from '@mui/x-data-grid';
import './MessageItem.css';

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

  // Convert dataframe to DataGrid format
  const getDataGridRows = (dataframe) => {
    if (!dataframe || dataframe.length === 0) return [];
    return dataframe.map((row, index) => ({
      id: index,
      ...row
    }));
  };

  // Generate columns for DataGrid
  const getDataGridColumns = (dataframe) => {
    if (!dataframe || dataframe.length === 0) return [];
    return Object.keys(dataframe[0]).map((key) => ({
      field: key,
      headerName: key,
      flex: 1,
      minWidth: 100,
      sortable: true,
      filterable: true,
    }));
  };

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
                  <CodeMirror
                    value={editedSql || message.sql}
                    onChange={setEditedSql}
                    height="200px"
                    extensions={[sql()]}
                    theme={githubLight}
                    basicSetup={{
                      lineNumbers: true,
                      highlightActiveLine: true,
                      highlightSelectionMatches: true,
                      autocompletion: true,
                      foldGutter: true,
                      indentOnInput: true,
                    }}
                    style={{ fontSize: '14px' }}
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
              <DataGrid
                rows={getDataGridRows(message.dataframe)}
                columns={getDataGridColumns(message.dataframe)}
                autoHeight
                density="compact"
                initialState={{
                  pagination: {
                    paginationModel: { pageSize: 5, page: 0 },
                  },
                  sorting: {
                    sortModel: [{ field: 'id', sort: 'asc' }],
                  },
                }}
                pageSizeOptions={[5, 10, 25, 50]}
                disableRowSelectionOnClick
                sx={{
                  '& .MuiDataGrid-cell': {
                    fontSize: '14px',
                    padding: '8px 16px',
                  },
                  '& .MuiDataGrid-columnHeaders': {
                    backgroundColor: '#f5f5f5',
                    fontWeight: 'bold',
                    fontSize: '14px',
                  },
                  '& .MuiDataGrid-row:hover': {
                    backgroundColor: '#f8f9fa',
                  },
                  '& .MuiDataGrid-footerContainer': {
                    borderTop: '1px solid #e0e0e0',
                  },
                  '& .MuiDataGrid-columnSeparator': {
                    color: '#e0e0e0',
                  },
                  '& .MuiDataGrid-cell:focus': {
                    outline: 'none',
                  },
                  '& .MuiDataGrid-columnHeader:focus': {
                    outline: 'none',
                  },
                }}
                componentsProps={{
                  pagination: {
                    labelRowsPerPage: 'Rows per page:',
                  },
                }}
              />
            </div>
          )}
        </div>
      </div>
    );
  }
  
  return null;
};

export default MessageItem; 