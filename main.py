from typing import Annotated, Optional
from typing_extensions import Doc
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse
from inference import main
import torch
import uvicorn
import os
import imageio
import shutil
from src.utils.videoio import load_video_to_cv2
from src.utils.face_enhancer import enhancer_generator_with_len

app = FastAPI()

class args:
    device: str
    driven_audio: str
    source_image: str 
    ref_eyeblink: Annotated[str|None, Doc("path to reference video providing eye blinking")] = None
    ref_pose: Annotated[str|None, Doc("path to reference video providing pose")] = None
    checkpoint_dir: Annotated[str, Doc("path to save models")] = './checkpoints'
    result_dir: Annotated[str, Doc("path to output")] = './temp'
    pose_style: Annotated[int, Doc("input pose style from [0, 46)")] = 0
    batch_size: Annotated[int, Doc("the batch size of facerender")] = 2
    size: Annotated[int, Doc("the image size of the facerender")] = 256
    expression_scale: Annotated[float, Doc("the batch size of facerender")] = 1.
    input_yaw: Annotated[int|None, Doc("the input yaw degree of the user")] = None
    input_pitch: Annotated[int|None, Doc("the input pitch degree of the user")] = None
    input_roll: Annotated[int|None, Doc("the input roll degree of the user")] = None
    enhancer: Annotated[str|None, Doc("Face enhancer, [gfpgan, RestoreFormer]")] = None
    background_enhancer: Annotated[str|None, Doc("background enhancer, [realesrgan]")] = None
    cpu: bool = False
    face3dvis: Annotated[bool, Doc("generate 3d face and 3d landmarks")] = False
    still: Annotated[bool, Doc("can crop back to the original videos for the full body aniamtion")] = True
    preprocess: Annotated[str, Doc("how to preprocess the images. choices=['crop', 'extcrop', 'resize', 'full', 'extfull']")] = 'crop'
    verbose: Annotated[bool, Doc("saving the intermedia output or not")] = True
    old_version: Annotated[bool, Doc("use the pth other than safetensor version")] = False

    # net structure and parameters
    init_path: Annotated[str|None, Doc("Useless")] = None
    use_last_fc: Annotated[bool, Doc("zero initialize the last fc")] = False
    net_recon: Annotated[str, Doc("useless. choices=['resnet18', 'resnet34', 'resnet50']")] = 'resnet50'
    bfm_folder: str = './checkpoints/BFM_Fitting/'
    bfm_model: Annotated[str, Doc("bfm model")] = 'BFM_model_front.mat'

    # default renderer parameters
    focal: float = 1015.
    center: float = 112.
    camera_d: float = 10.
    z_near: float = 5.
    z_far: float = 15.
    
@app.post("/test")
def myfunction(
    input_video: UploadFile,
    input_audio: UploadFile,
    enhancer: Annotated[str|None, Form()] = None,
    background_enhancer: Annotated[str|None, Form()] = None,
    preprocess: Annotated[str, Form()] = 'crop',
    still: Annotated[bool, Form()] = True,
    size: Annotated[int, Form()] = 256,
    ref_eyeblink: UploadFile|None = None,
    ref_pose: UploadFile|None = None
    ) -> FileResponse:
    video_path = f'temp/{input_video.filename}'
    audio_path = f'temp/{input_audio.filename}'
    
    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
    else:
        args.device = "cpu"
    
    with open(video_path,'wb+') as f:
        f.write(input_video.file.read())
        
    with open(audio_path,'wb+') as f:
        f.write(input_audio.file.read())
        
    args.preprocess = preprocess
    args.still = still
    args.size = size
        
    if ref_eyeblink:
        ref_eyeblink_path = 'temp/ref_eyeblink.mp4'
        with open(ref_eyeblink_path,'wb+') as f:
            f.write(ref_eyeblink.file.read())
        args.ref_eyeblink = ref_eyeblink_path
    
    if ref_pose:
        ref_pose_path = 'temp/ref_pose.mp4'
        with open(ref_pose_path,'wb+') as f:
            f.write(ref_pose.file.read())
        args.ref_pose = ref_pose_path
    
    args.driven_audio = audio_path
    args.source_image = video_path
    args.enhancer = enhancer
    args.background_enhancer = background_enhancer
    
    filepath = main(args)

    return FileResponse(filepath, media_type='video/mp4')

@app.post("/gfpgan")
def myfunction(input_video: UploadFile):
    if os.path.exists('temp'):
        shutil.rmtree('temp')
        print(f"All files in temp deleted.")
    os.mkdir('temp')
    
    video_path = f'temp/{input_video.filename}'
    with open(video_path,'wb+') as f:
        f.write(input_video.file.read())
    
    enhanced_images_gen_with_len = enhancer_generator_with_len(video_path)
    imageio.mimsave('temp/upscaled.mp4',enhanced_images_gen_with_len,  fps=float(25))
    
    return FileResponse('temp/upscaled.mp4')
    
if __name__ == "__main__":
    uvicorn.run(app)