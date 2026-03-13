<h1 align=center>
  Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos
</h1>

<p align=center>  
  Rohith Peddi, Shravan Shanmugam, Saurabh, Likhitha Pallapothula, Yu Xiang, Parag Singla, Vibhav Gogate
</p>

<p align="center">
    Under Review
</p>

<!-- <div align=center>
  <a src="https://img.shields.io/badge/project-website-green" href="">
    <img src="https://img.shields.io/badge/project-website-green">
  </a>
  <a src="https://img.shields.io/badge/paper-arxiv-red" href="">
    <img src="https://img.shields.io/badge/paper-arxiv-red">
  </a>
  <a src="https://img.shields.io/badge/bibtex-citation-blue" href="">
    <img src="https://img.shields.io/badge/bibtex-citation-blue">
  </a> 
</div> -->

<p align="center">
  (This page is under continuous update)
</p>

----

## UPDATE

- We shall release ActionGenome4D annotations, trained checkpoints and VLM evaluation code in June 2026

---- 

## Overview

### 1. World Scene Graph Generation (WSGG) Task
![WSGG Task Details](analysis/assets/WorldSGGTaskPicture.png)
The World Scene Graph Generation (WSGG) task involves predicting 3D bounding boxes and relationship properties (such as attention, spatial proximity, and contact) for objects within a continuous 4D scene setup.

### 2. 4D Scene Pipeline
![4D Scene Pipeline](analysis/assets/WorldSGG4DScenePipeline.png)
The 4D Scene Pipeline processes monocular video to construct a comprehensive 4D representation of the environment, integrating 3D object detection, tracking, and metric space projection across time.

### 3. ActionGenome4D Dataset
![Dataset Picture](analysis/assets/WorldSGGDatasetPicture.png)
An overview of the ActionGenome4D dataset, which provides rich 4D annotations for objects and their dynamic interactions over time across a variety of indoor environments.

### 4. Manual Relationship Correction
![Manual Relationship Correction Interface](analysis/assets/WorldSGGManualRelCorrection.png)
The Manual Relationship Correction interface allows for human-in-the-loop review and fine-grained modification of generated relationships, ensuring high-quality ground-truth annotations.

### 5. Manual 3D Floor Correction
![Manual 3D Floor Correction Interface](analysis/assets/WorldSGGManual3DFloorCorrection.png)
The Manual 3D Floor Correction tool provides a 3D annotation interface for aligning reconstructed point clouds with the ground plane. Through a multi-step process of rotation and translation adjustments, annotators correct the floor alignment to ensure accurate world-frame coordinate systems for all objects in the scene.

### 6. WSGG Model Pipeline
![WSGG Model Pipeline Architecture](analysis/assets/WorldSGGWSGG.png)
The WorldSGG architecture includes specialized encoders (structural, motion, camera pose), unobserved object representations (such as PWG, MWAE, and 4DST variants), and spatio-temporal decoders to predict complex object relationships in 4D.

### 7. MLLM Evaluation Pipeline
![MLLM Evaluation Pipeline](analysis/assets/WorldSGGMLLMPipeline.png)
The MLLM Pipeline utilizes Vision-Language Models to generate coarse event graphs and employs Large Language Models powered by Graph RAG to infer continuous world scene graphs from video segments.

-------
### ACKNOWLEDGEMENTS

This code is based on the following awesome repositories. 
We thank all the authors for releasing their code. 

- [Pi3](https://github.com/yyfz/Pi3)
- [PromptHMR](https://github.com/yufu-wang/PromptHMR)
- [Cut3R](https://cut3r.github.io/)
- [RAFT](https://github.com/princeton-vl/RAFT)
- [DepthAnything](https://github.com/DepthAnything/Depth-Anything-V2)
- [UniDepth](https://github.com/lpiccinelli-eth/UniDepth)
