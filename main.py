from fastapi import FastAPI
from transformers import AutoModel, AutoTokenizer
from pydantic import BaseModel
import torch

app = FastAPI()


class Texts(BaseModel):
    texts: list[str]


@app.post("/jina-embeddings")
async def jina_mbeddings(texts: Texts):
    model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True
    )
    embeddings = model.encode(texts.texts)
    return dict(embeddings=embeddings.tolist())


@app.post("/llm-embedder")
async def llm_embeddings(texts: Texts):
    tokenizer = AutoTokenizer.from_pretrained("BAAI/llm-embedder")
    model = AutoModel.from_pretrained("BAAI/llm-embedder", trust_remote_code=True)
    model.eval()

    encoded_input = tokenizer(
        texts.texts, padding=True, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        model_output = model(**encoded_input)
        embeddings = model_output[0][:, 0]
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return dict(embeddings=embeddings.tolist())
