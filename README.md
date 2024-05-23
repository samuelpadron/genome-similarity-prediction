# Predicting genome similarity with HyenaDNA

## HyenaDNA

![HyenaDNA_pipeline](assets/pipeline.png "HyenaDNA")

### Important links:  
- [arxiv](https://arxiv.org/abs/2306.15794)  
- [blog](https://hazyresearch.stanford.edu/blog/2023-06-29-hyena-dna)
- [colab](https://colab.research.google.com/drive/1wyVEQd4R3HYLTUOXEEQmp_I8aNC_aLhL?usp=sharing)  
- [huggingface](https://huggingface.co/LongSafari)
- [discord](https://discord.gg/RJxUq4mzmW)
- [youtube (talk)](https://youtu.be/haSkAC1fPX0?si=IUMmo_iGZ6SK1DBX)

Credit: The code of the backbone model comes from [HyenaDNA](https://github.com/HazyResearch/hyena-dna).

## Dependencies
<a name="dependencies"></a>

The repository is built using Pytorch Lightning and some other common libraries.

- clone repo, cd into it

- create a conda environment, with Python 3.8+

```
conda create -n hyena-dna python=3.8
```

- The repo is developed with Pytorch 1.13, using cuda 11.7

```
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```

- install requirements:
```
pip install -r requirements.txt
```

## Quick Entry point 

## Experiment