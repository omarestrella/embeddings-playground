# Getting Started

1. Create virtual environment (pyenv, python3 venv, etc)
2. Get dependencies: `pip install -r requirements.txt`
3. Run server: `uvicorn main:app --reload`
4. Make a request:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/jina-embeddings' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "texts": [
    "string"
  ]
}'

# Response: { embeddings: number[][] }
```
