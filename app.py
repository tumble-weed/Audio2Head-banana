from PIL import Image
import base64 
from io import BytesIO
import demo # this will be available from the Thin-Plate-Spline repo
import imageio
######################################################
# imports for animate
from skimage.transform import resize
import torch
from skimage import img_as_ubyte
from skimage.transform import resize
import os
import numpy as np
######################################################
# all settings, exposed here for visibility

config_path = 'config/vox-256.yaml'
checkpoint_path = 'checkpoints/vox.pth.tar'
device = 'cuda'
# driving_video='./assets/driving.mp4'
img_shape = (256,256)
find_best_frame = False
# choose from ['standard', 'relative', 'avd']
mode = 'standard'
######################################################
TODO = False
def decodeBase64Image(imageStr: str, name: str) -> Image:
    image = Image.open(BytesIO(base64.decodebytes(bytes(imageStr, "utf-8"))))
    print(f'Decoded image "{name}": {image.format} {image.width}x{image.height}')
    return image
def encodeBase64Image(image: Image) -> str:
    # https://stackoverflow.com/questions/31826335/how-to-convert-pil-image-image-object-to-base64-string
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str
def init():
    '''
    creates the globals that will be used every call
    '''
    # CUDA_VISIBLE_DEVICES=0 python demo.py --config config/vox-256.yaml --checkpoint checkpoints/vox.pth.tar --source_image ./source.jpg --driving_video ./driving.mp4
    global inpainting, kp_detector, dense_motion_network, avd_network
    inpainting, kp_detector, dense_motion_network, avd_network = demo.load_checkpoints(config_path = config_path, checkpoint_path = checkpoint_path, device = device)
    
def inference(all_inputs:dict) -> dict:
    # print('bounceback')
    # return all_inputs
    '''
    takes in dict created from request json, outputs dict
    to be wrapped up into a response json
    '''
    global inpainting, kp_detector, dense_motion_network, avd_network
    #==================================================================
    if 'image' not in all_inputs:
        return {'result':-1,'message':'image absent in request'}
    image = all_inputs.get("image", None)
    image = decodeBase64Image(image,'image')
    image = np.array(image)
    #==================================================================
    if 'video' not in all_inputs:
        return {'result':-1,'message':'video absent in request'}
    driving_video = all_inputs.get("video",None)
    driving_video = os.path.join('./videos',driving_video)
    if not os.path.exists(driving_video):
        return {'result':-1,'message':'video not recognized'}
    #==================================================================
    with torch.inference_mode():
        video_base64 = wrapper_for_animate(image,
                        driving_video=driving_video,
                        device='cuda',
                        img_shape = img_shape,
                        inpainting = inpainting, 
                        kp_detector  = kp_detector, 
                        dense_motion_network  = dense_motion_network, 
                        avd_network = avd_network,
                        find_best_frame = find_best_frame,
                        mode = mode,
                        # result_video='./result.mp4',
                        )
    #TODO: or storage to google bucket and send back the link?
    return {'result':video_base64,'message':'success'}
#######################################################################
# wrapper for animate
#######################################################################

def wrapper_for_animate(source_image,
                        driving_video,
                        device,
                        img_shape,
                        inpainting, 
                        kp_detector, 
                        dense_motion_network, 
                        avd_network,
                        find_best_frame,
                        mode,
                        # result_video='./result.mp4',
                        ):
    # source_image = imageio.imread(opt.source_image)
    reader = imageio.get_reader(driving_video)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()
    

    device = torch.device(device)
    source_image = resize(source_image, img_shape)[..., :3]
    driving_video = [resize(frame, img_shape)[..., :3] for frame in driving_video]
    #===============================================
    # copied from demo.py in Thin-Plate-Spline ...
    if find_best_frame:
        i = demo.find_best_frame(source_image, driving_video, False)
        print ("Best frame: " + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i+1)][::-1]
        predictions_forward = demo.make_animation(source_image, driving_forward, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = mode)
        predictions_backward = demo.make_animation(source_image, driving_backward, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = mode)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = demo.make_animation(source_image, driving_video, inpainting, kp_detector, dense_motion_network, avd_network, device = device, mode = mode)
    #===============================================
    # HACK: save result as temporary file,reread and binarize
    import tempfile
    temp_name = next(tempfile._get_candidate_names())
    temp_name = temp_name +'.mp4'
    imageio.mimsave(temp_name, [img_as_ubyte(frame) for frame in predictions], fps=fps)    
    # imageio.mimread(temp_name)
    # https://stackoverflow.com/questions/56248567/how-do-i-decode-encode-a-video-to-a-text-file-and-then-back-to-video
    with open(temp_name, "rb") as videoFile:
        video_base64 =  base64.b64encode(videoFile.read()).decode('utf-8')
    os.system(f'rm {temp_name}')
    #===============================================
    return video_base64
