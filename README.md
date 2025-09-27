<h1 align=center>
  Spatio Temporal Scene Graph Generation and Anticipation in Dynamic 4D Scenes
</h1>

<p align=center>  
  Rohith Peddi, Saurabh, Harsh Garg, Parag Singla, Vibhav Gogate
</p>

<p align="center">
    Under Review
</p>

<div align=center>
  <a src="https://img.shields.io/badge/project-website-green" href="https://rohithpeddi.github.io/#/impartail">
    <img src="https://img.shields.io/badge/project-website-green">
  </a>
  <a src="https://img.shields.io/badge/paper-arxiv-red" href="https://arxiv.org/pdf/2403.04899v1.pdf">
    <img src="https://img.shields.io/badge/paper-arxiv-red">
  </a>
  <a src="https://img.shields.io/badge/bibtex-citation-blue" href="">
    <img src="https://img.shields.io/badge/bibtex-citation-blue">
  </a> 
</div>

<p align="center">
  (This page is under continuous update)
</p>

----

## UPDATE


-------
### ACKNOWLEDGEMENTS

This code is based on the following awesome repositories. 
We thank all the authors for releasing their code. 


-------
# SETUP

## Environment Setup

```bash
conda create -n scene4cast python=3.12 pip

conda activate scene4cast


pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```




## Dataset Preparation 

------

# Instructions to run

Please see the scripts/training for Python modules.

Please see the scripts/tests for testing Python modules.

------


### Notes

To remove .DS_Store files from a directory, run the following command in the terminal:

```bash
find . -name .DS_Store -print0 | xargs -0 git rm -f --ignore-unmatch
``` 
