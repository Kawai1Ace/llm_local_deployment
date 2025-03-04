import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

import torch
import uvicorn

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from threading import Thread

from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

@asynccontextmanager
async def lifespan(app: FastAPI): # collects GPU memory
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

with open('index.html') as f:
    html = f.read()

@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            json_request = await websocket.receive_json()
            query = json_request['query']
            history = json_request['history']
            show_content = ""
            for response in qwen_stream_chat(model, tokenizer, query, history):
                show_content += response
                await websocket.send_json({'response': show_content, 'history': history, 'status': 202})
            await websocket.send_json({'status': 200})
            show_content = ""
    except WebSocketDisconnect:
        pass



def qwen_stream_chat(model, tokenizer, query, history):
    conversation = history
    conversation.append({'role': 'user', 'content': query})
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors='pt',
    )
    inputs = inputs.to(model.device)
    streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True)
    generation_kwargs = dict(
        input_ids=inputs,
        streamer=streamer,
        max_new_tokens = 1024
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    for new_text in streamer:
        yield new_text

if __name__ == "__main__":
    model_name_or_path = "Your model name or path"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).cuda()
    model.eval()
    
    uvicorn.run(f"{__name__}:app", host='0.0.0.0', port=12231, workers=1)