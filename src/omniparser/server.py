import sys
import os
import time
import base64
from fastapi import FastAPI
from fastapi.responses import Response
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
    'device': os.getenv('DEVICE', 'cuda'),
    'BOX_TRESHOLD': float(os.getenv('BOX_THRESHOLD', '0.05')),
    'host': os.getenv('OMNIPARSER_HOST', 'localhost'),
    'port': int(os.getenv('OMNIPARSER_PORT', '8081'))
}

app = FastAPI()
omniparser = Omniparser(config)

class ParseRequest(BaseModel):
    base64_image: str

@app.post("/parse/")
async def parse(parse_request: ParseRequest):
    # print('start parsing...')
    start = time.time()
    dino_labled_img, parsed_content_list = omniparser.parse(parse_request.base64_image)
    latency = time.time() - start
    # print('time:', latency)
    return {"som_image_base64": dino_labled_img, "parsed_content_list": parsed_content_list, 'latency': latency}

@app.get("/test_origianl_image/")
async def test_org_image():
    test_image_path = os.path.join('test_image', 'windows_desktop.png')
    with open(test_image_path, "rb") as image_file:
        image_bytes = image_file.read()
    return Response(content=image_bytes, media_type="image/jpeg")


@app.get("/probe/")
async def root():
    return {"message": "Omniparser API ready"}

@app.get("/test_result_image/")
async def test_result_iumage():
    test_image_path = os.path.join('test_image', 'windows_desktop.png')
    with open(test_image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        dino_labled_img, _ = omniparser.parse(base64_image)
        image_bytes = base64.b64decode(dino_labled_img)
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