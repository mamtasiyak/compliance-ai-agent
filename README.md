# Enterprise Agent Copilot

AI powered backend system for extracting structured compliance obligations from unstructured documents.

## Overview

This project builds a production-ready AI backend that:

* Accepts raw document text via API
* Uses LLMs to extract compliance obligations
* Validates outputs using strict schemas
* Returns structured, machine-readable results

## Key Features

* FastAPI-based backend
* Pydantic v2 schema validation
* Modular service layer architecture
* LLM abstraction (provider-agnostic design)
* Fault-tolerant parsing and validation
* Structured AI output (no raw text responses)

## Architecture

API Layer → Service Layer → LLM Provider → Validation → Structured Output

```
app/
  schemas/        # Data models (Pydantic)
  services/       # Business logic
  main.py         # FastAPI entrypoint
tests/            # Test files
```

## Tech Stack

* Python 3.12
* FastAPI
* Pydantic v2
* OpenAI (LLM integration)
* uv (dependency management)

## Environment Setup

Create a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

## Running the Project

```bash
uv venv
source .venv/bin/activate
uv sync
uvicorn app.main:app --reload
```

Visit:

```
http://127.0.0.1:8000/docs
```