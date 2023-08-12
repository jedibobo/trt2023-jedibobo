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

from utilities import Engine,device_view
from cuda import cudart
from polygraphy import cuda
class hackathon:
    def initialize(self):
        self.apply_canny = CannyDetector()
        self.model = create_model("./models/cldm_v15.yaml").cpu()
        self.model.load_state_dict(
            load_state_dict(
                "/home/player/ControlNet/models/control_sd15_canny.pth", location="cuda"
            )
        )
        self.model = self.model.cuda()
        self.ddim_sampler = DDIMSampler(self.model)

        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.trt_logger, "")

        self.state_dict = {
            "clip": "cond_stage_model",
            "control_net": "control_model",
            "unet": "diffusion_model",
            "vae": "first_stage_model",
        }
        self.acc_clip_stage = True
        self.model.acc_control_stage = True
        self.model.acc_unet_stage = True
        self.acc_vae_stage = True

        self.clip_engine = None
        self.model.control_engine = None
        self.model.unet_engine = None
        self.vae_engine = None

        _, free_mem, _ = cudart.cudaMemGetInfo()
        GiB = 2 ** 30
        if free_mem > 6*GiB:
            activation_carveout = 4*GiB
            self.max_workspace_size = free_mem - activation_carveout
        else:
            self.max_workspace_size = 0
        
        self.stream = cuda.Stream()
        self.model.stream = cuda.Stream()

        if not os.path.exists("./engine/"):
            os.makedirs("./engine/")
            
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
                if not os.path.isfile("engine/sd_clip_transformer_fp16.engine") and self.acc_clip_stage:
                    prompt = "a bird"
                    a_prompt = "best quality, extremely detailed"
                    n_prompt = ("longbody, lowres, bad anatomy, bad hands, missing fingers")
                    num_samples = 1
                    text = [prompt + ", " + a_prompt] * num_samples
                    batch_encoding = self.tokenizer(
                        text,
                        truncation=True,
                        max_length=77,
                        return_length=True,
                        return_overflowing_tokens=False,
                        padding="max_length",
                        return_tensors="pt",
                    )
                    # tokens = batch_encoding["input_ids"].to("cuda")
                    # outputs = model(input_ids=tokens, output_hidden_states="last"=="hidden")'
                    with torch.inference_mode(), torch.autocast("cuda"):
                        inputs = torch.zeros(2, 77, dtype=torch.int32).to("cuda")
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
                            input_names=[
                                "input_ids",
                            ],
                            output_names=[
                                "last_hidden_state",
                                "pooler_output",
                            ],
                            dynamic_axes={
                                "input_ids": {0: "B"},
                                "last_hidden_state": {0: "B"},
                                "pooler_output": {0: "B"},
                            },
                        )
                    
                    os.system("python3 modify_clip_transformer_onnx.py")
                    os.system("polygraphy surgeon sanitize sd_clip_transformer_reshape.onnx --fold-constants --override-input-shapes 'input_ids:[2,77]' -o sd_clip_transformer_sanitize.onnx")
                    os.system("polygraphy surgeon extract sd_clip_transformer_sanitize.onnx --inputs input_ids:[2,77]:int32 --outputs last_hidden_state:float32 -o sd_clip_subgraph.onnx")
                    
                    clip_engine = Engine("engine/sd_clip_transformer_fp16.engine")
                    clip_profile = {'input_ids':[(2, 77),(2, 77),(2, 77)]}
                    clip_engine.build("sd_clip_subgraph.onnx",fp16=True,input_profile=clip_profile,enable_all_tactics=True,workspace_size=self.max_workspace_size)
                    if model is not None:
                        del model
                        del clip_engine
                        torch.cuda.empty_cache()
                        gc.collect()

                    # os.system("trtexec --onnx=sd_clip_subgraph.onnx --saveEngine=engine/sd_clip_transformer_fp16.engine --fp16 --builderOptimizationLevel=5")
                # print("engine exists", os.path.exists("./sd_clip_transformer_fp16.engine"))
                print("finished converting clip model")
            # 再导出control_net
            elif k == "control_net" and self.model.acc_control_stage:
                H = 256
                W = 384
                # control_net默认就是forwrad,不需要做什么操作了
                control_model = self.model.control_model
                if not os.path.isfile("engine/sd_control_fp16.engine"):
                    with torch.inference_mode(), torch.autocast("cuda"):
                        x_in = torch.randn(1, 4, H // 8, W // 8, dtype=torch.float32).to(
                            "cuda"
                        )
                        h_in = torch.randn(
                            1, 3, H, W, dtype=torch.float32).to("cuda")
                        t_in = torch.zeros(1, dtype=torch.int32).to("cuda")
                        c_in = torch.randn(
                            1, 77, 768, dtype=torch.float32).to("cuda")

                        controls = control_model(
                            x=x_in, hint=h_in, timesteps=t_in, context=c_in
                        )

                        output_names = []
                        for i in range(13):
                            output_names.append("out_" + str(i))

                        dynamic_table = {
                            "x_in": {0: "bs", 2: "H", 3: "W"},
                            "h_in": {0: "bs", 2: "8H", 3: "8W"},
                            "t_in": {0: "bs"},
                            "c_in": {0: "bs"},
                        }

                        for i in range(13):
                            dynamic_table[output_names[i]] = {0: "bs"}

                        torch.onnx.export(
                            control_model,
                            (x_in, h_in, t_in, c_in),
                            "./sd_control.onnx",
                            export_params=True,
                            opset_version=17,
                            do_constant_folding=True,
                            keep_initializers_as_inputs=True,
                            input_names=["x_in", "h_in", "t_in", "c_in"],
                            output_names=output_names,
                            dynamic_axes=dynamic_table,
                        )
                    os.system("polygraphy surgeon sanitize sd_control.onnx --fold-constants --override-input-shapes 'x_in:[1,4,32,48]' 'h_in:[1,3,256,384]' 't_in:[1,]' 'c_in:[1,77,768]' -o sd_control_sanitize.onnx")
                    # os.system("trtexec --onnx=sd_control_sanitize.onnx --saveEngine=engine/sd_control_fp16.engine --fp16 --builderOptimizationLevel=5")
                    control_engine = Engine("engine/sd_control_fp16.engine")
                    control_profile = {'x_in':[(1,4,32,48),(1,4,32,48),(1,4,32,48)],'h_in':[(1,3,256,384),(1,3,256,384),(1,3,256,384)],'t_in':[(1,),(1,),(1,)],'c_in':[(1,77,768),(1,77,768),(1,77,768)]} 
                    control_engine.build("sd_control_sanitize.onnx",fp16=True,input_profile=control_profile,enable_all_tactics=True,workspace_size=self.max_workspace_size)
                    if control_model is not None:
                        del control_model
                        del control_engine
                        torch.cuda.empty_cache()
                        gc.collect()
                print("finished converting control model")
                # 下面的和上面一样了，输入输出可以去通过debug一下`cldm/cldm.py`下面的`apply_model`函数，这里打一下断点，你就知道有几个输入和输出了。
                # 输出是一个包含13个torch.Tensor的list,所以output_names也应该是一个13长度的List
                # 导出的时候会报错，提示有个算子不支持，直接将`models/cldm_v15.yaml`文件里面的`use_checkpoint`设置False，对于controlNet和Unet都是这样设置的
            # 再导出unet
            elif k == "unet" and self.model.acc_unet_stage:
                # import ipdb; ipdb.set_trace()
                H = 256
                W = 384
                model = self.model.model.diffusion_model
                if not os.path.isfile("engine/sd_unet_fp16.engine"):
                    with torch.inference_mode(), torch.autocast("cuda"):
                        x_in = torch.randn(1, 4, H // 8, W // 8,
                                        dtype=torch.float32).to("cuda")
                        t_in = torch.zeros(1, dtype=torch.int32).to("cuda")
                        c_in = torch.randn(
                            1, 77, 768, dtype=torch.float32).to("cuda")
                        con_in = []
                        h = 32
                        w = 48
                        # prepare inputs
                        control_0 = torch.randn(
                            1, 320, h, w, dtype=torch.float32).to("cuda")
                        control_1 = torch.randn(
                            1, 320, h, w, dtype=torch.float32).to("cuda")
                        control_2 = torch.randn(
                            1, 320, h, w, dtype=torch.float32).to("cuda")
                        control_3 = torch.randn(
                            1, 320, h // 2, w // 2, dtype=torch.float32).to("cuda")
                        control_4 = torch.randn(
                            1, 640, h // 2, w // 2, dtype=torch.float32).to("cuda")
                        control_5 = torch.randn(
                            1, 640, h // 2, w // 2, dtype=torch.float32).to("cuda")
                        control_6 = torch.randn(
                            1, 640, h // 4, w // 4, dtype=torch.float32).to("cuda")
                        control_7 = torch.randn(
                            1, 1280, h // 4, w // 4, dtype=torch.float32).to("cuda")
                        control_8 = torch.randn(
                            1, 1280, h // 4, w // 4, dtype=torch.float32).to("cuda")
                        control_9 = torch.randn(
                            1, 1280, h // 8, w // 8, dtype=torch.float32).to("cuda")
                        control_10 = torch.randn(
                            1, 1280, h // 8, w // 8, dtype=torch.float32).to("cuda")
                        control_11 = torch.randn(
                            1, 1280, h // 8, w // 8, dtype=torch.float32).to("cuda")
                        control_12 = torch.randn(
                            1, 1280, h // 8, w // 8, dtype=torch.float32).to("cuda")
                        con_in = [control_0, control_1, control_2, control_3, control_4, control_5,
                                control_6, control_7, control_8, control_9, control_10, control_11, control_12,]

                        con_in_name = []
                        for i in range(13):
                            con_in_name.append("control_in_" + str(i))
                        eps = model(x=x_in,
                                    timesteps=t_in,
                                    context=c_in,
                                    control=con_in,
                                    only_mid_control=False,
                                    )
                        output_names = ["diffution_model_output"]
                        dynamic_table = {
                            "x_in": {0: "bs"},
                            "t_in": {0: "bs"},
                            "c_in": {0: "bs"},
                            "diffution_model_output": {0: "bs"},
                        }

                        torch.onnx.export(
                            model,
                            (
                                x_in,
                                t_in,
                                c_in,
                                [control_0,
                                    control_1,
                                    control_2,
                                    control_3,
                                    control_4,
                                    control_5,
                                    control_6,
                                    control_7,
                                    control_8,
                                    control_9,
                                    control_10,
                                    control_11,
                                    control_12,]
                            ),
                            "unet-onnx/sd_unet.onnx",
                            export_params=True,
                            opset_version=17,
                            do_constant_folding=True,
                            keep_initializers_as_inputs=True,
                            input_names=["x_in", "t_in", "c_in"] + con_in_name,
                            output_names=output_names,
                            dynamic_axes=dynamic_table,
                        )
                    # import ipdb;ipdb.set_trace()
                    # os.system(
                    #     " polygraphy surgeon sanitize unet-onnx/sd_unet.onnx --fold-constants --override-input-shapes 'x_in:[1,4,32,48]' 't_in:[1,]' 'c_in:[1,77,768]' 'control_in_0:[1,320,32,48]' 'control_in_1:[1,320,32,48]' 'control_in_2:[1,320,32,48]' 'control_in_3:[1,320,16,24]' 'control_in_4:[1,640,16,24]' 'control_in_5:[1x640,16,24]' 'control_in_6:[1,640,8,12' 'control_in_7:[1,1280,8,12]' 'control_in_8:[1,1280,8,12]' 'control_in_9:[1,1280,4,6]' 'control_in_10:[1,1280,4,6]' 'control_in_11:[1,1280,4,6]' 'control_in_12:[1,1280,4,6]' -o unet-onnx/sd_unet_sanitize.onnx"
                    # )
                    unet_engine = Engine("engine/sd_unet_fp16.engine")
                    unet_profile = {'x_in':[(1,4,32,48),(1,4,32,48),(1,4,32,48)],
                                    't_in':[(1,),(1,),(1,)],
                                    'c_in':[(1,77,768),(1,77,768),(1,77,768)],
                                    'control_in_0':[(1,320,32,48),(1,320,32,48),(1,320,32,48)],
                                    'control_in_1':[(1,320,32,48),(1,320,32,48),(1,320,32,48)],
                                    'control_in_2':[(1,320,32,48),(1,320,32,48),(1,320,32,48)],
                                    'control_in_3':[(1,320,16,24),(1,320,16,24),(1,320,16,24)],
                                    'control_in_4':[(1,640,16,24),(1,640,16,24),(1,640,16,24)],
                                    'control_in_5':[(1,640,16,24),(1,640,16,24),(1,640,16,24)],
                                    'control_in_6':[(1,640,8,12),(1,640,8,12),(1,640,8,12)],
                                    'control_in_7':[(1,1280,8,12),(1,1280,8,12),(1,1280,8,12)],
                                    'control_in_8':[(1,1280,8,12),(1,1280,8,12),(1,1280,8,12)],
                                    'control_in_9':[(1,1280,4,6),(1,1280,4,6),(1,1280,4,6)],
                                    'control_in_10':[(1,1280,4,6),(1,1280,4,6),(1,1280,4,6)],
                                    'control_in_11':[(1,1280,4,6),(1,1280,4,6),(1,1280,4,6)],
                                    'control_in_12':[(1,1280,4,6),(1,1280,4,6),(1,1280,4,6)],}
                    unet_engine.build("unet-onnx/sd_unet.onnx",fp16=True,input_profile=unet_profile,enable_all_tactics=True,workspace_size=self.max_workspace_size)
                    # os.system("trtexec --onnx=unet-onnx/sd_unet.onnx  --saveEngine=engine/sd_unet_fp16.engine --fp16 --builderOptimizationLevel=5 --optShapes=x_in:1x4x32x48,t_in:1,c_in:1x77x768,control_in_0:1x320x32x48,control_in_1:1x320x32x48,control_in_2:1x320x32x48,control_in_3:1x320x16x24,control_in_4:1x640x16x24,control_in_5:1x640x16x24,control_in_6:1x640x8x12,control_in_7:1x1280x8x12,control_in_8:1x1280x8x12,control_in_9:1x1280x4x6,control_in_10:1x1280x4x6,control_in_11:1x1280x4x6,control_in_12:1x1280x4x6")
                    if model is not None:
                        del model
                        del unet_engine
                        torch.cuda.empty_cache()
                        gc.collect()
                print("finished converting unet model")
                # 和control_net一样的操作，一样的bug,就是输入输出有些不一样
                # unet输入包含controlNet的输出，所以input_names里面。由于controlNet的输出是一个长度为13的List，所以这个input_names,需要将对应变量的输入名，改成13个，例如control_1, control_2..control_13这样。
            # 再导出vae
            elif k == "vae" and self.acc_vae_stage:
                pass
                model = temp_model
                # vae调用的是decode,而导出onnx需要forward,所以这里做一个替换即可。
                model.forward = model.decode
                h = 32
                w = 48
                # control_net默认就是forwrad,不需要做什么操作了
                if not os.path.isfile("engine/sd_vae_fp16.engine"):
                    with torch.inference_mode(), torch.autocast("cuda"):
                        z_in = torch.randn(
                            1, 4, h, w, dtype=torch.float32).to("cuda")
                        output_names = ['vae_decode_out']
                        dynamic_table = {
                            "z_in": {0: "bs"},
                            "vae_decode_out": {0: "bs"},
                        }
                        torch.onnx.export(
                            model,
                            z_in,
                            "./sd_vae.onnx",
                            export_params=True,
                            opset_version=17,
                            do_constant_folding=True,
                            keep_initializers_as_inputs=True,
                            input_names=["z_in"],
                            output_names=output_names,
                            dynamic_axes=dynamic_table,
                        )
                    os.system("polygraphy surgeon sanitize sd_vae.onnx --fold-constants --override-input-shapes 'z_in:[1,4,32,48]'  -o sd_vae_sanitize.onnx")
                    vae_engine = Engine("engine/sd_vae_fp16.engine")
                    vae_profile = {'z_in':[(1,4,32,48),(1,4,32,48),(1,4,32,48)]} 
                    vae_engine.build("sd_vae_sanitize.onnx",fp16=True,input_profile=vae_profile,enable_all_tactics=True,workspace_size=self.max_workspace_size)
                    # os.system("trtexec --onnx=sd_vae_sanitize.onnx --saveEngine=engine/sd_vae_fp16.engine --fp16 --builderOptimizationLevel=5")
                    if model is not None:
                        del model
                        del vae_engine
                        torch.cuda.empty_cache()
                        gc.collect()
                print("finished converting vae model")
                # # 然后和上面一样，做onnx导出
            else:
                model = None

        # # remove model from gpu
        # if model is not None:
        #     del model
        #     torch.cuda.empty_cache()
        #     gc.collect()
        if self.acc_clip_stage:
            self.clip_engine = Engine("engine/sd_clip_transformer_fp16.engine")
            self.clip_engine.load()
            self.clip_engine.activate()
            clip_profile = {'input_ids':(2, 77)}
            self.clip_engine.allocate_buffers(clip_profile)

        if self.model.acc_control_stage:
            self.model.control_engine = Engine("engine/sd_control_fp16.engine")
            self.model.control_engine.load()
            control_profile = {'x_in':(1,4,32,48),'h_in':(1,3,256,384),'t_in':(1,),'c_in':(1,77,768)} 
            self.model.control_engine.activate()
            self.model.control_engine.allocate_buffers()

        if self.model.acc_unet_stage:
            self.model.unet_engine = Engine("engine/sd_unet_fp16.engine")
            self.model.unet_engine.load()
            self.model.unet_engine.activate()
            unet_profile = {'x_in':(1,4,32,48),
                            't_in':(1,),
                            'c_in':(1,77,768),
                            'control_in_0': (1,320,32,48),
                            'control_in_1': (1,320,32,48),
                            'control_in_2': (1,320,32,48),
                            'control_in_3': (1,320,16,24),
                            'control_in_4': (1,640,16,24),
                            'control_in_5': (1,640,16,24),
                            'control_in_6': (1,640,8,12),
                            'control_in_7': (1,1280,8,12),
                            'control_in_8': (1,1280,8,12),
                            'control_in_9': (1,1280,4,6),
                            'control_in_10':(1,1280,4,6),
                            'control_in_11':(1,1280,4,6),
                            'control_in_12':(1,1280,4,6)}
            self.model.unet_engine.allocate_buffers()
        if self.acc_vae_stage:    
            self.vae_engine = Engine("engine/sd_vae_fp16.engine")
            self.vae_engine.load()
            self.vae_engine.activate()
            vae_profile = {'z_in':(1,4,32,48)} 
            self.vae_engine.allocate_buffers(vae_profile)

    def process(
        self,
        input_image,
        prompt,
        a_prompt,
        n_prompt,
        num_samples,
        image_resolution,
        ddim_steps,
        guess_mode,
        strength,
        scale,
        seed,
        eta,
        low_threshold,
        high_threshold,
    ):
        with torch.no_grad():
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            detected_map = self.apply_canny(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)

            control = torch.from_numpy(
                detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, "b h w c -> b c h w").clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            if self.acc_clip_stage:
                text = [prompt + ", " + a_prompt] * num_samples+ [n_prompt] * num_samples
                batch_encoding = self.tokenizer(
                        text,
                        truncation=True,
                        max_length=77,
                        return_length=True,
                        return_overflowing_tokens=False,
                        padding="max_length",
                        return_tensors="pt",
                    )
                tokens = batch_encoding["input_ids"].type(torch.int32).to("cuda")
                clip_transformer_out = self.clip_engine.infer({"input_ids": device_view(tokens.to(dtype=torch.int32))}, self.stream)['last_hidden_state']

                cond = {"c_concat": [control],"c_crossattn": [clip_transformer_out[0:1]],}
                un_cond = {"c_concat": None if guess_mode else [control],"c_crossattn": [clip_transformer_out[1:2]],}

            else:
                cond = {
                    "c_concat": [control],
                    "c_crossattn": [
                        self.model.get_learned_conditioning(
                            [prompt + ", " + a_prompt] * num_samples
                        )
                    ],
                }
                un_cond = {
                    "c_concat": None if guess_mode else [control],
                    "c_crossattn": [
                        self.model.get_learned_conditioning(
                            [n_prompt] * num_samples)
                    ],
                }

            shape = (4, H // 8, W // 8)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=True)

            self.model.control_scales = (
                [strength * (0.825 ** float(12 - i)) for i in range(13)]
                if guess_mode
                else ([strength] * 13)
            )  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = self.ddim_sampler.sample(
                12,
                num_samples,
                shape,
                cond,
                verbose=False,
                eta=eta,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=un_cond,
            )

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)
            # import ipdb; ipdb.set_trace()
            if self.acc_vae_stage:
                scale_factor = 0.18215
                samples = 1. / scale_factor * samples
                x_samples = self.vae_engine.infer({"z_in": device_view(samples.to(dtype=torch.float32))}, self.stream)['vae_decode_out']
            else:
                x_samples = self.model.decode_first_stage(samples)
            x_samples = (
                (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
                .cpu()
                .numpy()
                .clip(0, 255)
                .astype(np.uint8)
            )

            results = [x_samples[i] for i in range(num_samples)]
        return results

if __name__  == "__main__":
    hk = hackathon()
    hk.initialize()