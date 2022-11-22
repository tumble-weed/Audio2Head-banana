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
import yaml
import argparse
######################################################
from modules.audio2pose import get_pose_from_audio
from skimage import io, img_as_float32
import cv2
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from modules.audio2kp import AudioModel3D
import yaml,os,imageio
import subprocess
from inference import draw_annotation_box,get_audio_feature_from_audio
import tempfile
######################################################
TODO = None
######################################################
# all settings, exposed here for visibility
'''
config_path = 'config/vox-256.yaml'
'''
config_file = "config/vox-256.yaml"
parameters_file = "config/parameters.yaml"
model_path = 'checkpoints/audio2head.pth.tar'

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
    
    # https://github.com/wangsuzhen/Audio2Head/blob/09e9b431e48a6358c2877a12cd45457ff0379455/inference.py#L153
    
    global kp_detector,generator,audio2kp,opt

    with open(config_file) as f:
        config = yaml.load(f)
    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    kp_detector = kp_detector.cuda()
    generator = generator.cuda()

    opt = argparse.Namespace(**yaml.load(open(parameters_file)))
    audio2kp = AudioModel3D(opt).cuda()

    checkpoint  = torch.load(model_path)
    kp_detector.load_state_dict(checkpoint["kp_detector"])
    generator.load_state_dict(checkpoint["generator"])
    audio2kp.load_state_dict(checkpoint["audio2kp"])

    generator.eval()
    kp_detector.eval()
    audio2kp.eval()
    

def inference(all_inputs:dict) -> dict:
    # print('bounceback')
    # return all_inputs
    '''
    takes in dict created from request json, outputs dict
    to be wrapped up into a response json
    '''
    # return {'result':-1,'message':'success'}                    
    global kp_detector,generator,audio2kp,opt
    #==================================================================
    # https://github.com/wangsuzhen/Audio2Head/blob/09e9b431e48a6358c2877a12cd45457ff0379455/inference.py#L121
    if 'audio' not in all_inputs:
        return {'result':-1,'message':'audio missing'}
    if 'image' not in all_inputs:
        return {'result':-1,'message':'image missing'}

    image = all_inputs.get("image", None)
    image = decodeBase64Image(image,'image')
    img = image = np.array(image)
    img = cv2.resize(img, (256, 256))
    img = np.array(img_as_float32(img))
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).cuda()
    # return {'result':-1,'message':'success'}                    
    #====================================================================
    # https://stackoverflow.com/questions/50279380/how-to-decode-base64-string-directly-to-binary-audio-format

    audio_path = next(tempfile._get_candidate_names())  + ".wav"
    wav_file = open(audio_path, "wb")
    audio = all_inputs.get("audio", None)
    decode_string = base64.b64decode(audio)
    wav_file.write(decode_string)

    # temp_audio="./results/temp.wav"
    temp_audio="temp.wav"
    command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (audio_path, temp_audio))
    output = subprocess.call(command, shell=True, stdout=None)
    # return {'result':-1,'message':'success'}                    
    #====================================================================
    audio_feature = get_audio_feature_from_audio(temp_audio)
    frames = len(audio_feature) // 4

    ref_pose_rot, ref_pose_trans = get_pose_from_audio(img, audio_feature, model_path)
    torch.cuda.empty_cache()
    #==================================================================
    # return {'result':-1,'message':'success'}                    
    with torch.inference_mode():
        video_base64 = wrapper_for_inference(
                    opt,
                    img,
                    audio_path,
                    frames,
                    audio_feature,
                    ref_pose_trans,
                    ref_pose_rot,
                    kp_detector,
                    audio2kp,
                    generator,
                    )
    os.system(f'rm {audio_path}')
    print('wrapper_for_inference done')
    return {'result':video_base64,'message':'success'}                    
#######################################################################
# wrapper for inference
#######################################################################
def wrapper_for_inference(
                    opt,
                    img,
                    audio_path,
                    frames,
                    audio_feature,
                    ref_pose_trans,
                    ref_pose_rot,
                    kp_detector,
                    audio2kp,
                    generator,
                    ):
    audio_f = []
    poses = []
    pad = np.zeros((4,41),dtype=np.float32)
    for i in range(0, frames, opt.seq_len // 2):
        temp_audio = []
        temp_pos = []
        for j in range(opt.seq_len):
            if i + j < frames:
                temp_audio.append(audio_feature[(i+j)*4:(i+j)*4+4])
                trans = ref_pose_trans[i + j]
                rot = ref_pose_rot[i + j]
            else:
                temp_audio.append(pad)
                trans = ref_pose_trans[-1]
                rot = ref_pose_rot[-1]

            pose = np.zeros([256, 256])
            draw_annotation_box(pose, np.array(rot), np.array(trans))
            temp_pos.append(pose)
        audio_f.append(temp_audio)
        poses.append(temp_pos)
    print('loop1 done')
    audio_f = torch.from_numpy(np.array(audio_f,dtype=np.float32)).unsqueeze(0)
    poses = torch.from_numpy(np.array(poses, dtype=np.float32)).unsqueeze(0)

    bs = audio_f.shape[1]
    predictions_gen = []
    total_frames = 0
    
    for bs_idx in range(bs):
        t = {}

        t["audio"] = audio_f[:, bs_idx].cuda()
        t["pose"] = poses[:, bs_idx].cuda()
        t["id_img"] = img
        kp_gen_source = kp_detector(img)

        gen_kp = audio2kp(t)
        if bs_idx == 0:
            startid = 0
            end_id = opt.seq_len // 4 * 3
        else:
            startid = opt.seq_len // 4
            end_id = opt.seq_len // 4 * 3

        for frame_bs_idx in range(startid, end_id):
            tt = {}
            tt["value"] = gen_kp["value"][:, frame_bs_idx]
            if opt.estimate_jacobian:
                tt["jacobian"] = gen_kp["jacobian"][:, frame_bs_idx]
            out_gen = generator(img, kp_source=kp_gen_source, kp_driving=tt)
            out_gen["kp_source"] = kp_gen_source
            out_gen["kp_driving"] = tt
            del out_gen['sparse_deformed']
            del out_gen['occlusion_map']
            del out_gen['deformed']
            predictions_gen.append(
                (np.transpose(out_gen['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0] * 255).astype(np.uint8))

            total_frames += 1
            if total_frames >= frames:
                break
        if total_frames >= frames:
            break
    print('loop2 done')
    with tempfile.TemporaryDirectory() as log_dir:
        # path = os.path.join(tmp, 'something')
        # use path
        # log_dir = save_path
        if not os.path.exists(os.path.join(log_dir, "temp")):
            os.makedirs(os.path.join(log_dir, "temp"))
        temp_name = next(tempfile._get_candidate_names())        
        image_name = temp_name + ".mp4"

        video_path = os.path.join(log_dir, "temp", image_name)

        imageio.mimsave(video_path, predictions_gen, fps=25.0)

        save_video = os.path.join(log_dir, image_name)
        cmd = r'ffmpeg -y -i "%s" -i "%s" -vcodec copy "%s"' % (video_path, audio_path, save_video)
        os.system(cmd)
        os.remove(video_path)    
        # https://stackoverflow.com/questions/56248567/how-do-i-decode-encode-a-video-to-a-text-file-and-then-back-to-video
        with open(save_video, "rb") as videoFile:
            video_base64 =  base64.b64encode(videoFile.read()).decode('utf-8')
        os.system(f'rm {save_video}')
        #===============================================
    print('video64 done')
    return video_base64

