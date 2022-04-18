# Argument Extraction by Generation

This repository contains the code of the paper titled "EA$^2$E: Improving Consistency with Event Awareness for Document-level Argument Extraction" accpeted in Findings of the Annual Conference of the North American Chapter of the Association for Computational Linguistics: NAACL 2022.

## Set up

conda create --name eaae python=3.9 
conda activate eaae
pip install -r requirement.txt
python -m spacy download en_core_web_sm

## Run
To run the EA^2E model, use sh scripts/train_eaae.sh
To test the model, use sh scripts/test_bart_gen.sh


## Datasets
- ACE05 (Access from LDC[https://catalog.ldc.upenn.edu/LDC2006T06] and preprocessing following OneIE[http://blender.cs.illinois.edu/software/oneie/])
