from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import os
import gc
import tensorrt as trt

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from ldm.util import log_txt_as_img, exists, instantiate_from_config


class hackathon():

    def initialize(self):
        self.apply_canny = CannyDetector()
        self.model = create_model('./models/cldm_v15.yaml').cpu()
        self.model.load_state_dict(load_state_dict('/home/player/ControlNet/models/control_sd15_canny.pth', location='cuda'))
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)
        

        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.trt_logger, '')
        
        self.state_dict = {
            "clip": "cond_stage_model",
            "control_net": "control_model",
            "unet": "diffusion_model",
            "vae": "first_stage_model"
        }
        self.acc_clip_stage = False
        self.acc_control_stage = True
        self.acc_unet_stage = True
        self.acc_vae_stage = True

        # 先导出clip
        for k, v in self.state_dict.items():
            if k != "unet":
                temp_model = getattr(self.model, v)
            else:
            # 注意unet比较特殊，他是在self.model.model下面的一个diffusion_model属性里面
                temp_model = getattr(self.model.model, v)
            if k == "clip":
                # 因为clip调用了encode,encode调用了forward,forward输入时，一共就两步，一个是tokenizer,一个是transformer，transformer的__call__默认也是forward,所以干脆我直接去clip的transformer导出onnx就行了。
                model = temp_model.transformer
                # 这里把tokenizer单独拿出来做一个函数，后续给prompt做分词用
                self.tokenizer = temp_model.tokenizer
                # 然后下面就是onnx导出代码了，基本老一套
                # 下面的inputs是输入变量列表，你可以去debug一下，实际是一个[torch.Tensor]
                # import ipdb; ipdb.set_trace()
                if not os.path.isfile("sd_clip_transformer.engine"):
                    prompt = "a bird"
                    a_prompt = "best quality, extremely detailed"
                    n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers"
                    num_samples = 1
                    text = [prompt + ', ' + a_prompt] * num_samples
                    batch_encoding = self.tokenizer(text, truncation=True, max_length=77, return_length=True,
                                            return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
                    tokens = batch_encoding["input_ids"].to("cuda")
                    # outputs = model(input_ids=tokens, output_hidden_states="last"=="hidden")
                    inputs = torch.zeros(1,77, dtype=torch.int32).to("cuda")
                    # import ipdb; ipdb.set_trace()
                    # onnx_path: 保存路径
                    # input_names, output_names 对应输入输出名列表，这个debug一下，可以随便命名
                    # dynamic_axes是一个dict，这里我给一个样例：{'input_ids': {0: 'B'},'text_embeddings': {0: 'B'}}，大概意思就是输入变量`input_ids`和输出变量`text_embeddings`的第0维可以动态变化，其实也就是batch_size支持动态咯。
                    # opset_version: 这里固定为17就行
                    torch.onnx.export(
                        model,
                        inputs,
                        "./sd_clip_transformer.onnx",
                        export_params=True,
                        opset_version=17,
                        do_constant_folding=True,
                        input_names=['input_ids',],
                        output_names=['last_hidden_state',],
                        dynamic_axes={'input_ids': {0: 'B'},'last_hidden_state': {0: 'B'}},
                    )
                    
                    os.system("python3 modify_clip_transformer_onnx_shape.py")
                    os.system("polygraphy surgeon sanitize sd_clip_transformer_onnx_reshape.onnx --fold-constants --override-input-shapes 'input_ids:[1,77]' -o sd_clip_transformer_sanitize.onnx")
                    os.system("polygraphy surgeon extract sd_clip_transformer_sanitize.onnx --inputs input_ids:[1,77]:int32 --outputs last_hidden_state:float32 -o sd_clip_subgraph.onnx")
                    os.system("trtexec --onnx=sd_clip_subgraph.onnx --saveEngine=sd_clip_transformer_fp16.engine --fp16 ")

                print("engine exists", os.path.exists("./sd_clip_transformer_fp16.engine"))

                with open("./sd_clip_transformer_fp16.engine", 'rb') as f:
                    engine_str = f.read()

                clip_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_str)
                clip_context = clip_engine.create_execution_context()
                clip_context.set_binding_shape(0, (1, 77))
                self.clip_context = clip_context  # TODO modify this to support inner acquire this function
                print("finished converting clip model")               
            # 再导出control_net
            elif k == 'control_net':
                H = 256
                W = 384
                # control_net默认就是forwrad,不需要做什么操作了
                model = temp_model
                control_model = self.model.control_model
                if not os.path.isfile("sd_control_fp16.engine"):
                    x_in = torch.randn(1, 4, H//8, W //8, dtype=torch.float32).to("cuda")
                    h_in = torch.randn(1, 3, H, W, dtype=torch.float32).to("cuda")
                    t_in = torch.zeros(1, dtype=torch.int64).to("cuda")
                    c_in = torch.randn(1, 77, 768, dtype=torch.float32).to("cuda")

                    controls = control_model(x=x_in, hint=h_in, timesteps=t_in, context=c_in)

                    output_names = []
                    for i in range(13):
                        output_names.append("out_"+ str(i))

                    dynamic_table = {'x_in' : {0 : 'bs', 2 : 'H', 3 : 'W'}, 
                                        'h_in' : {0 : 'bs', 2 : '8H', 3 : '8W'}, 
                                        't_in' : {0 : 'bs'},
                                        'c_in' : {0 : 'bs'}}
                    
                    for i in range(13):
                        dynamic_table[output_names[i]] = {0 : "bs"}

                    torch.onnx.export(control_model,               
                                        (x_in, h_in, t_in, c_in),  
                                        "./sd_control_test.onnx",   
                                        export_params=True,
                                        opset_version=17,
                                        do_constant_folding=True,
                                        keep_initializers_as_inputs=True,
                                        input_names = ['x_in', "h_in", "t_in", "c_in"], 
                                        output_names = output_names, 
                                        dynamic_axes = dynamic_table)
                    
                    os.system("trtexec --onnx=sd_control_test.onnx --saveEngine=sd_control_fp16.engine --fp16 --optShapes=x_in:1x4x32x48,h_in:1x3x256x384,t_in:1,c_in:1x77x768")

                with open("./sd_control_fp16.engine", 'rb') as f:
                    engine_str = f.read()

                control_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_str)
                control_context = control_engine.create_execution_context()

                control_context.set_binding_shape(0, (1, 4, H // 8, W // 8))
                control_context.set_binding_shape(1, (1, 3, H, W))
                control_context.set_binding_shape(2, (1,))
                control_context.set_binding_shape(3, (1, 77, 768))
                self.model.control_context = control_context

                print("finished converting control model")
                # 下面的和上面一样了，输入输出可以去通过debug一下`cldm/cldm.py`下面的`apply_model`函数，这里打一下断点，你就知道有几个输入和输出了。
                # 输出是一个包含13个torch.Tensor的list,所以output_names也应该是一个13长度的List
                # 导出的时候会报错，提示有个算子不支持，直接将`models/cldm_v15.yaml`文件里面的`use_checkpoint`设置False，对于controlNet和Unet都是这样设置的
            # 再导出unet
            elif k == "unet":
                model = temp_model
                # 和control_net一样的操作，一样的bug,就是输入输出有些不一样
                # unet输入包含controlNet的输出，所以input_names里面。由于controlNet的输出是一个长度为13的List，所以这个input_names,需要将对应变量的输入名，改成13个，例如control_1, control_2..control_13这样。
            # 再导出vae
            elif k == "vae":
                model = temp_model
                # vae调用的是decode,而导出onnx需要forward,所以这里做一个替换即可。
                model.forward = model.decode
                # 然后和上面一样，做onnx导出
            else:
                model = None
        


        # remove model from gpu
        if model is not None:
            del model
            torch.cuda.empty_cache()
            gc.collect()



    def process(self, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
        with torch.no_grad():
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            detected_map = self.apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            if self.acc_clip_stage:
                cond = {}
                un_cond = {}
                for i in range(2):
                    if i == 0:
                        text = [prompt + ', ' + a_prompt] * num_samples
                    elif i == 1:
                        text = [n_prompt] * num_samples
                    else:
                        raise NotImplementedError
                    text = [prompt + ', ' + a_prompt] * num_samples
                    batch_encoding = self.tokenizer(text, truncation=True, max_length=77, return_length=True,
                            return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
                    tokens = batch_encoding["input_ids"].to("cuda")
                    buffer_device = []
                    buffer_device.append(tokens.reshape(-1).data_ptr())
                    
                    clip_transformer_out = []
                    temp = torch.zeros(1, 77, 768, dtype=torch.float32).to("cuda")
                    clip_transformer_out.append(temp)
                    buffer_device.append(temp.reshape(-1).data_ptr())

                    temp = torch.zeros(1, 768, dtype=torch.float32).to("cuda")
                    clip_transformer_out.append(temp)
                    buffer_device.append(temp.reshape(-1).data_ptr())
                    
                    self.clip_context.execute_v2(buffer_device)
                    if i == 0:
                        cond["c_concat"] = [control]
                        cond["c_crossattn"] = [clip_transformer_out[0]]
                    elif i == 1:
                        un_cond["c_concat"] = None if guess_mode else [control]
                        un_cond["c_crossattn"] = [clip_transformer_out[0]]
            else:    
                cond = {"c_concat": [control], "c_crossattn": [self.model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
                un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)]}
            

            # import ipdb; ipdb.set_trace()
            
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=True)

            self.model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = self.ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(num_samples)]
        return results