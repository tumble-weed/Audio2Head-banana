import skimage.io
import base64
from PIL import Image
from io import BytesIO
import app
import numpy as np
def encodeBase64Image(image: Image) -> str:
    # https://stackoverflow.com/questions/31826335/how-to-convert-pil-image-image-object-to-base64-string
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str
def decodeBase64Image(imageStr: str, name: str) -> Image:
    image = Image.open(BytesIO(base64.decodebytes(bytes(imageStr, "utf-8"))))
    print(f'Decoded image "{name}": {image.format} {image.width}x{image.height}')
    return image
image  = skimage.io.imread('assets/source.png')
image = Image.fromarray(image)
image_base64 = encodeBase64Image(image)
# request.json({'image':image_base64})

if 'bounce back experiment' and False:
    app.init()
    response = app.inference({'image':image_base64})
    image = decodeBase64Image(response['original_image'], 'bounce-back-experiment')
    image = np.array(image)
    skimage.io.imsave('bounce_back_image.png',image)
    
if 'run dl code' and True:
    app.init()
    if 'correct':
        response = app.inference({'image':image_base64,'video':'driving.mp4'})
        video_base64 = response['result']
        with open('result.mp4','wb') as f:
            f.write(base64.b64decode(video_base64))
        # image = np.array(image)
        # skimage.io.imsave('bounce_back_image.png',image)
    if 'missing image':
        response = app.inference({'video':'driving.mp4'})
        print('missing image')
        print(response)
    elif 'missing video':
        response = app.inference({'image':image_base64})
        print('missing video')
        print(response) 
    elif 'wrong video':
        response = app.inference({'image':image_base64,'video':'driving0.mp4'})
        print('wrong video')
        print(response)         