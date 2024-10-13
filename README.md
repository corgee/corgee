<h1 align="center">Contrastive Generalizable Embeddings</h1>
<hr>

<div><h3 align="center">
    <img width="320" height="200" src="resources/logo.png" />
</h3></div>
![alternative text](logo.png)

Highly optimized implementation of CORGEE to train state-of-the-art embedding models.

# Prepare data and config
You can first check and evaluate available models before you train/ fine-tune your own model.
To train from scratch/ fine-tune an available encoder (recommended) you need to select a suitable encoder from the available ones/ prepare training data and modify training config. More details on this can be found here.

# Submit training jobs
These steps assume that you have setup training data and training config in above section. Also you need to have access to a Singularity VC or have a local GPU server.
## Using Æther
- Upload config file to a path in cosmos
- Clone the pipeline: aether://TODO. Some template pipelines are available here: TODO
- Configure the following parameters: Username, PAT, Config path, Output directory
- Aether runs also convert the final model to ONNX format for faster inference
## Using Amulet
- Clone repository
- Setup Amulet (https://amulet-docs.azurewebsites.net/main/index.html)
- Run the interactive submission script: source remote/amlt_submit.sh
## Locally on GPU servers
- Clone repository
- Setup packages remote/setup.sh
- Start training: run.sh

# Inference in Aether
- Embedding inference using AdsBrain based inference
- Sample recall evaluation pipeline

# Available models
TODO: provide embedding inference pipelines and evaluation pipelines for each of these
- General models: 
  - English
  - Multilingual
- Retail click models:
  - English
  - Multilingual INTL
  - Multilingual global
- General retail models:
  - English
  - Multilingual INTL
  - Multilingual global

# Mainstreamings
Refer here for a summary of product impact from models trained by CORGEE

<hr>
May thy models converge, and generalize better
