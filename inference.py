import os
import argparse
import torch

from src.pipeline_tavid import TAViDPipeline
from diffusers import CogVideoXDPMScheduler
from diffusers.utils import export_to_video, load_image


def main():
    parser = argparse.ArgumentParser(description="Generate video using CogVideoX and TAViD LoRA weights")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the image")
    parser.add_argument("--mask_path", type=str, required=True, help="Path to the mask")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints", help="Path to the LoRA weights")
    parser.add_argument("--prompt", type=str, default="", help="Prompt for the video")
    parser.add_argument("--output_file", type=str, default="output.mp4", help="Output video file")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CogVideoX pipeline
    pipe = TAViDPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16).to(device)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)

    # Increase the number of input channels of the image projection layer for the mask input
    # and load the weights
    with torch.no_grad():
        new_conv_in = torch.nn.Conv2d( # 1 for mask
            pipe.transformer.patch_embed.proj.in_channels + 1, pipe.transformer.patch_embed.proj.out_channels, \
                pipe.transformer.patch_embed.proj.kernel_size, pipe.transformer.patch_embed.proj.stride, pipe.transformer.patch_embed.proj.padding
        )
        pipe.transformer.patch_embed.proj = new_conv_in
    pipe.transformer.patch_embed.proj.load_state_dict(torch.load(os.path.join(args.ckpt_path, "patch_embed_conv2d.pth"), weights_only=True))
    pipe.transformer.patch_embed.proj.to(dtype=torch.bfloat16, device=device)

    # Load TAViD LoRA weights
    pipe.load_lora_weights(args.ckpt_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="spatial_lora")
    pipe.set_adapters(["spatial_lora"], [1.0])

    # If you have enough VRAM (27GB+), comment out the following lines
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    # generate video
    video = pipe(
                image=load_image(args.image_path),
                mask=load_image(args.mask_path),
                prompt=args.prompt,
                use_dynamic_cfg=True,
            ).frames[0]
    
    os.makedirs("results", exist_ok=True)
    export_to_video(video, os.path.join("results", args.output_file), fps=8)
    print(f"Video saved")

if __name__ == "__main__":
    main()



