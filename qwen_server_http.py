import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import time
import torch
import uvicorn

from torch.nn import Module
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict, List, Literal, Optional, Union
from sse_starlette.sse import ServerSentEvent, EventSourceResponse

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from rich import print as rprint

@asynccontextmanager
async def lifespan(app: FastAPI):
    '''异步上下文管理器 lifespan，用于清零CUDA缓存'''

    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache() #来清空 CUDA 缓存，释放 GPU 内存
        torch.cuda.ipc_collect()

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_length: Optional[int] = None
    stream: Optional[bool] = False

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]

class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]

class CompletionUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    usage: Optional[CompletionUsage] = None
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    global model_args
    model_card = ModelCard(id="qwen1.5-1.8b")
    llama_card = ModelCard(id="llama3.2-1b")
    return ModelList(data=[model_card, llama_card])

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer, llama_model, llama_tokenizer

    # 限制只能用户身份提问
    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    
    messages = request.messages
    
    if request.model == "qwen1.5-1.8b":
        process_model = model
        process_tokenizer = tokenizer
    elif request.model == "llama3.2-1b":
        process_model = llama_model
        process_tokenizer = llama_tokenizer
    else:
        raise HTTPException(status_code=400, detail="Invalid request")


    text = process_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = process_tokenizer([text], return_tensors="pt").to("cuda")
    generated_ids = process_model.generate(
        model_inputs.input_ids, 
        max_new_tokens=request.max_length if request.max_length is not None else 2048, 
        temperature=request.temperature if request.temperature is not None else 0.7, 
        top_p=request.top_p if request.top_p is not None else 0.95
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = process_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=ChatMessage(role="assistant", content=response),
        finish_reason="stop"
    )

    message = request.messages
    prompt_tokens = len(process_tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True))
    completion_tokens = len(process_tokenizer(response).input_ids)
    total_tokens = prompt_tokens + completion_tokens
    usage = CompletionUsage(
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        total_tokens=total_tokens
    )
    return ChatCompletionResponse(
        model=request.model,
        choices=[choice_data],
        usage=usage,
        object="chat.completion"
    )

    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )
    # model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    # generated_ids = model.generate(
    #     model_inputs.input_ids, 
    #     max_new_tokens=request.max_length if request.max_length is not None else 2048, 
    #     temperature=request.temperature if request.temperature is not None else 0.7, 
    #     top_p=request.top_p if request.top_p is not None else 0.95
    # )
    # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # choice_data = ChatCompletionResponseChoice(
    #     index=0,
    #     message=ChatMessage(role="assistant", content=response),
    #     finish_reason="stop"
    # )

    # message = request.messages
    # prompt_tokens = len(tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True))
    # completion_tokens = len(tokenizer(response).input_ids)
    # total_tokens = prompt_tokens + completion_tokens
    # usage = CompletionUsage(
    #     completion_tokens=completion_tokens,
    #     prompt_tokens=prompt_tokens,
    #     total_tokens=total_tokens
    # )
    # return ChatCompletionResponse(
    #     model=request.model,
    #     choices=[choice_data],
    #     usage=usage,
    #     object="chat.completion"
    # )



if __name__ == "__main__":
    model_name_or_path = "Your model name or path"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).cuda()

    llama_model_name_or_path = "Your model name or path"
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_name_or_path, padding_side="left")
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_model = AutoModelForCausalLM.from_pretrained(llama_model_name_or_path).cuda()

    model.eval()
    uvicorn.run(app, host="0.0.0.0", port=12356, workers=1)