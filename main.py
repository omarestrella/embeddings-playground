from fastapi import FastAPI
from transformers import AutoModel
from pydantic import BaseModel

app = FastAPI()


class Texts(BaseModel):
    texts: list[str]


@app.post("/jina-embeddings")
async def embeddings(texts: Texts):
    model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v2-base-en", trust_remote_code=True
    )
    embeddings = model.encode(texts.texts)
    return dict(embeddings=embeddings.tolist())
