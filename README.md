# Anjaylytics Backend

This repository contains the FastAPI backend for the Anjaylytics app. It provides endpoints for daily investment recommendations, metrics, reliability, trade export, and guides for Botswana and global investing. See `anjaylytics_full_api_v11.py` for details.

## Running locally

Install dependencies:

```
pip install -r requirements.txt
```

Run the server:

```
uvicorn anjaylytics_full_api_v11:app --reload --port 8080
```
