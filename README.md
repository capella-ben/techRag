# TechRAG Application

## Overview

The TechRAG application is a versatile RAG tool with an interactive chatbot interface. It provides functionalities for ingesting data from URLs and manually entered facts. The application is built using LangGraph, Milvus and Gradio.

## Features

### Chatbot Interface

The RAG-based chatbot interface allows users to interact with the application by sending messages and receiving responses. Leveraging Retrieval-Augmented Generation (RAG), the chatbot effectively handles a history of previous interactions, providing a more contextually aware and coherent conversation experience.

The LangGraph application defines a workflow using a state graph that includes nodes for web search, document retrieval, document grading, and content generation. The workflow starts with retrieving documents, followed by grading them. Depending on the grading results, the workflow either performs a web search or proceeds to generate content.

### URL Ingestion

Users can submit a list of URLs to the application for ingestion. This feature processes the content from the provided URLs and incorporates it into the application's knowledge base. This functionality is useful for dynamically updating the information the chatbot can access.

### Fact Ingestion

In addition to URL ingestion, users can manually enter facts into the system. This feature requires a title, description, and the fact itself, allowing for detailed and structured data entry.


## Requirements

1. Install libraries as per requirements.txt
2. A Milvus vector store.  
  1. Authentication i not required
3. API keys for:
  1. OpenAi
  2. Tavily


## Project Configuration

This project relies on a set of environment variables to configure various aspects of its functionality. These variables are stored in a `.env` file. Below is a detailed explanation of each variable and its purpose.

### Environment Variables

#### Database Configuration

- **VECTOR_DB_STORE**: The IP address of the vector database store.
  - Example: `192.168.1.100`

#### User Agent

- **USER_AGENT**: The user agent string to be used for web requests.
  - Example: `techRag`

#### API Keys

- **OPENAI_API_KEY**: The API key for accessing OpenAI services.
  - Example: `sk-1234567890abcdef1234567890abcdef`

- **TAVILY_API_KEY**: The API key for accessing Tavily services.
  - Example: `tb-0987654321fedcba0987654321fedcba`

#### Tracing Configuration (optional)

- **LANGCHAIN_TRACING_V2**: Enables or disables LangChain tracing. Set to `false` to disable.
  - Example: `false`

- **LANGCHAIN_ENDPOINT**: The endpoint URL for LangChain API.
  - Example: `https://api.smith.langchain.com`

- **LANGCHAIN_API_KEY**: The API key for accessing LangChain services.
  - Example: `lk-11223344556677889900aabbccddeeff`

### Usage

1. Create a `.env` file in the root directory of your project.
2. Copy and paste the environment variables listed above into the `.env` file.
3. Replace the example values with your actual configuration values.
4. Ensure your application is configured to load environment variables from the `.env` file.

### Example `.env` File

```env
VECTOR_DB_STORE=192.168.1.100
USER_AGENT=techRag

OPENAI_API_KEY=sk-1234567890abcdef1234567890abcdef
TAVILY_API_KEY=tb-0987654321fedcba0987654321fedcba

# Tracing (optional)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=lk-11223344556677889900aabbccddeeff
```

### Notes

- The **VECTOR_DB_STORE** variable should be set to the IP address of your vector database store.
- The **USER_AGENT** should be a string that represents your application.
- The **OPENAI_API_KEY** and **TAVILY_API_KEY** should be obtained from their respective service providers.
- The tracing configuration is optional and can be omitted if not needed.

