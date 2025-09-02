## Get Started
1. Install ROSE Dependencies

   You can follow the [Dependencies and Installation](https://github.com/Kunbyte-AI/ROSE?tab=readme-ov-file#dependencies-and-installation).

3. Install Demo Dependencies
```shell
cd hugging_face

# install python dependencies 
pip3 install -r requirements.txt

# Run the demo
python app.py
```
*Note: if you have problems downloading "git+https", you can set up SSH in Github and then replace them with "git+ssh://git@github.com/..."*

## Usage Guidance
* Step 1: Upload your video and click the `Get video info` button.


* Step 2: 
   1. *[Optional]* Specify the tracking period for the currently added mask by dragging the `Track start frame` or `Track end frame`.
   2. Click the image on the left to select the mask area.
   3. - Click `Add mask` if you are satisfied with the mask, or
      - *[Optional]* Click `Clear clicks` if you want to reselect the mask area, or
      - *[Optional]* Click `Remove mask` to remove all masks.
   4. *[Optional]* Go back to step 2.1 to add another mask.

   
* Step 3: 
   1. Click the `Tracking` button to track the masks for the whole video.
   2. Then click `Inpainting` to get the inpainting results.


*You can always refer to the `Highlighted Text` box on the page for guidance on the next step!*


## Citation
If you find our repo useful for your research, please consider citing our paper:
```bibtex
@article{miao2025rose,
      title={ROSE: Remove Objects with Side Effects in Videos}, 
      author={Miao, Chenxuan and Feng, Yutong and Zeng, Jianshu and Gao, Zixiang and Liu, Hantang and Yan, Yunfeng and Qi, Donglian and Chen, Xi and Wang, Bin and Zhao, Hengshuang},
      journal={arXiv preprint arXiv:2508.18633},
      year={2025}
}
```



## Acknowledgements

The project harnesses the capabilities from [Track Anything](https://github.com/gaomingqi/Track-Anything), [Segment Anything](https://github.com/facebookresearch/segment-anything) and [Cutie](https://github.com/hkchengrex/Cutie). Also the gradio demo page is based on [ProPainter's huggingface demo page](https://github.com/sczhou/ProPainter/tree/main/web-demos/hugging_face). Thanks for their awesome works.
