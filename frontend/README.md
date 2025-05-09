# Duckchat React App

This is the React frontend for Duckchat, a data querying application that allows you to use natural language to generate SQL queries and visualize results.

## Features

- Upload CSV and Parquet files
- Generate SQL queries using natural language prompts
- Edit and re-run SQL queries
- Visualize query results as tables
- Support for various LLM providers (OpenAI, Ollama)

## Development

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm start
```

The app will be available at http://localhost:3000

### Building for Production

```bash
# Create production build
npm run build
```

## Docker Deployment

The application can be run using Docker and docker-compose:

```bash
# From the project root (parent directory)
docker-compose up -d
```

This will build and start the React app along with the API backend and Ollama service.

## Architecture

- React frontend for user interface
- FastAPI backend for data processing and SQL generation
- DuckDB for database operations
- LLM integrations for natural language processing 