import sys
import os
import time
import base64
from fastapi import FastAPI
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from dotenv import load_dotenv 
import uvicorn
from util.omniparser import Omniparser
import asyncio
# # logging 
# import logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

# Configuration from environment variables with defaults
config = {
    'som_model_path': os.getenv('SOM_MODEL_PATH', 'weights/icon_detect/model.pt'),
    'caption_model_name': os.getenv('CAPTION_MODEL_NAME', 'florence2'),
    'caption_model_path': os.getenv('CAPTION_MODEL_PATH', 'weights/icon_caption_florence'),
    'device': os.getenv('DEVICE', 'auto'),
    'BOX_TRESHOLD': float(os.getenv('BOX_THRESHOLD', '0.05')),
    'host': os.getenv('OMNIPARSER_HOST', 'localhost'),
    'port': int(os.getenv('OMNIPARSER_PORT', '8081'))
}

app = FastAPI()
omniparser = Omniparser(config)

# Cap concurrent inference (GPU memory) and shed load with 429 + Retry-After when saturated.
MAX_CONCURRENCY = max(1, int(os.getenv('OMNIPARSER_MAX_CONCURRENCY', '1')))
ACQUIRE_TIMEOUT = float(os.getenv('OMNIPARSER_ACQUIRE_TIMEOUT_SECONDS', '5'))
RETRY_AFTER_SECONDS = int(os.getenv('OMNIPARSER_RETRY_AFTER_SECONDS', '5'))
inference_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

class ParseRequest(BaseModel):
    base64_image: str

async def run_inference(base64_image: str):
    """Run omniparser.parse in a worker thread; returns None if no slot frees up in time."""
    try:
        await asyncio.wait_for(inference_semaphore.acquire(), timeout=ACQUIRE_TIMEOUT)
    except asyncio.TimeoutError:
        return None
    try:
        return await asyncio.to_thread(omniparser.parse, base64_image)
    finally:
        inference_semaphore.release()

def busy_response():
    return JSONResponse(
        status_code=429,
        content={"detail": "Omniparser is busy, retry later"},
        headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
    )

@app.post("/parse/")
async def parse(parse_request: ParseRequest):
    start = time.time()
    result = await run_inference(parse_request.base64_image)
    if result is None:
        return busy_response()
    dino_labled_img, parsed_content_list = result
    latency = time.time() - start
    return {"som_image_base64": dino_labled_img, "parsed_content_list": parsed_content_list, 'latency': latency}

@app.get("/test_origianl_image/")
async def test_org_image():
    test_image_path = os.path.join('test_image', 'windows_desktop.png')
    with open(test_image_path, "rb") as image_file:
        image_bytes = image_file.read()
    return Response(content=image_bytes, media_type="image/jpeg")


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/probe/")
async def root():
    return {"message": "Omniparser API ready"}

@app.get("/test_result_image/")
async def test_result_iumage():
    test_image_path = os.path.join('test_image', 'windows_desktop.png')
    with open(test_image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    result = await run_inference(base64_image)
    if result is None:
        return busy_response()
    image_bytes = base64.b64decode(result[0])
    return Response(content=image_bytes, media_type="image/png")


if __name__ == "__main__":
    # # Run the async initialization
    # loop = asyncio.get_event_loop()
    # init_success = loop.run_until_complete(test_result_iumage())
    # if not init_success:
    #     print("Failed to initialize Omniparser.")
    #     sys.exit(1)
    # else:
    #     print("Omniparser initialized successfully.")
    uvicorn.run("server:app", port=config['port'], reload=True)