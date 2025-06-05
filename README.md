# rdp-video-surveillance

## Run OmniParser (Florence-2) model using Docker

 - OmniParser is going to detect icons in the given image 
 - Details can be found [here](https://github.com/microsoft/OmniParser)

**Build docker**

```cmd
# Basic build command
docker build -t rdp-video-surveillance-omniparser -f src/omniparser/Dockerfile src/omniparser
```

**Run docker**

 - Use GPU for the best performance

```cmd
docker run --gpus all -p 8000:8000 rdp-video-surveillance-omniparser
```

**Test**

```cmd
python test/test.py
```



