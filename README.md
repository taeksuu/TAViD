<p align="center">
  <h1 align="center">Target Aware Video Diffusion Models</h1>
  <p align="center">
    arXiv 2025
  </p>
  <p align="center">
    <a href="https://taeksuu.github.io/">Taeksoo Kim</a>,
    <a href="https://jhugestar.github.io/">Hanbyul Joo</a>
  </p>
  <p align="center">
    <a href="https://arxiv.org/pdf/2503.18950">
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://taeksuu.github.io/TAViD/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
    </a>
    <a href='https://huggingface.co/Taeksoo/TAViD' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Model-Hugging%20Face-yellow?style=flat&logo=Hugging%20face&logoColor=yellow' alt='Model Hugging Face'>
    </a>
  </p>
</p>

<p align="center">
    <img src="assets/demo.gif" alt="Logo" width="190%">
    <b>TL;DR. We use the segmentation mask as an additional input to specify the target for video diffusion mdoels.</b>
</p>

## Updates
- [2025/05/20] Initial code for inference and checkpoint released.
- [2025/03/25] Paper released.


TODO list.
- [ ] Release training code and dataset
- [ ] Add attention visualization code
- [ ] Add Gradio app

## Table of Contents
<ul>
    <li>
        <a href="#installation">Installation</a>
    </li>
    <li>
        <a href="#inference">Inference</a>
    </li>
    <li>
        <a href="#training and dataset">Training and Dataset</a>
    </li>
    <li>
        <a href="#citation">Citation</a>
    </li>
    <li>
        <a href="#acknowledgement">Acknowledgement</a>
    </li>
    <!-- <li>
        <a href="#license">License</a>
    </li> -->
</ul>

## Installation
Clone the repository.
```bash
git clone https://github.com/taeksuu/TAViD.git
```

Create a conda environment and install the required packages.
```bash
conda create -n tavid python=3.11
conda activate tavid
pip install -r requirements.txt
```
Download the model checkpoints and place it in the `./checkpoints` folder.
```bash
huggingface-cli download Taeksoo/TAViD --local-dir checkpoints
```

## Inference
You can run the inference code with the following command. Just add the trigger word "target" in front of the noun you'd like to specify as in the example below. You will find your output videos in the `./results` folder. \
We have tested the inference code on RTX 3090 and A100.
```bash
python inference.py \
  --image_path assets/image.png \
  --mask_path assets/mask_0.png \
  --prompt "In a serene, well-lit kitchen with clean, modern lines, the woman reaches forward, and picks up the target mug cup with her hand. She brings the target mug to her lips, taking a slow, thoughtful sip of the coffee, her gaze unfocused as if lost in contemplation. The steam from the coffee curls gently in the air, adding warmth to the quiet ambiance of the room."
```
Since our base model, CogvideoX, is trained with long prompts, prompt quality directly impacts the output quality. Please refer to <a href="https://github.com/THUDM/CogVideo/blob/main/inference/convert_demo.py">this guide</a> from CogVideoX for prompt enhancement. The generated videos can still suffer from limitations, including object disappearances or implausible dynamics. You may have to try multiple times for the best results.

## Training and Dataset
We will soon release the training code and data.


## Citation
If you find TAViD useful for your work, please consider citing:
```bibtex
@article{kim2025target,
    title={Target-Aware Video Diffusion Models},
    author={Kim, Taeksoo and Joo, Hanbyul},
    journal={arXiv preprint arXiv:2503.18950},
    year={2025}
}
```


## Acknowledgements
We sincerely thank the authors of following amazing works for their open-sourced codes, models, and datasets:
- Training: [CogVideo](https://github.com/THUDM/CogVideo), [Finetrainers](https://github.com/a-r-r-o-w/finetrainers)
- Datasets: [Ego-Exo4D](https://ego-exo4d-data.org/), [BEHAVE](https://virtualhumans.mpi-inf.mpg.de/behave/)


<!-- ## License
Codes are available only for non-commercial research purposes. -->