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

### Document Ingestion

MS Word document can be ingested using this feature.  They are split along the lines of headers so it is reommended that these are represented in the document. Headings levels 1 and 2 are split. 


## Requirements

1. Install pandoc: https://pandoc.org/
  1. Must be version 3.2 or higher
  2. Download from: https://github.com/jgm/pandoc/releases/latest
  3. Install via: https://pandoc.org/installing.html
2. Install libraries as per requirements.txt
3. A Milvus vector store.  
  1. Authentication i not required
4. API keys for:
  1. OpenAi
  2. Tavily


## Project Configuration

The following provides details on configuration required for the project.

### Environment Variables

This project relies on a set of environment variables to configure various aspects of its functionality. These variables are stored in a `.env` file. Below is a detailed explanation of each variable and its purpose.

#### Database Configuration

- **VECTOR_DB_STORE**: The IP address of the Milvus vector database store.
  - Example: `192.168.1.100`

- **COLLECTION_NAME**: The name of the collection in the Milvus vector store.
  - Example: `techRag`

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

#### Usage

1. Create a `.env` file in the root directory of your project.
2. Copy and paste the environment variables listed above into the `.env` file.
3. Replace the example values with your actual configuration values.
4. Ensure your application is configured to load environment variables from the `.env` file.

#### Example `.env` File

```env
VECTOR_DB_STORE=192.168.1.100
COLLECTION_NAME=techRag
USER_AGENT=techRag

OPENAI_API_KEY=sk-1234567890abcdef1234567890abcdef
TAVILY_API_KEY=tb-0987654321fedcba0987654321fedcba

# Tracing (optional)
LANGCHAIN_TRACING_V2=false
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=lk-11223344556677889900aabbccddeeff
```

### Milvus Collection Setup

A collection must be created in the Milvus database.  Once the .env file has been setup run the following script:
```Milvus_Create_Collection.py```


## Startup

Once the requirements are met and the configuration made, the application can be started by executing:
```app.py```

