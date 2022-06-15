This repository contains the code of the paper titled ["EA$^2$E: Improving Consistency with Event Awareness for Document-level Argument Extraction"](https://blender.cs.illinois.edu/paper/docie2022b.pdf) accpeted in Findings of the Annual Conference of the North American Chapter of the Association for Computational Linguistics: NAACL 2022.

## Code
Our code is based on [this repository](https://github.com/raspberryice/gen-arg).
```
./data: dataset
./docs: some documents
./scripts: running scripts
./src: all codes
```


## Set up
```sh
git clone https://github.com/ZQS1943/DOCIE.git
cd DOCIE
conda create --name eaae python=3.9 
conda activate eaae
pip install -r requirement.txt
python -m spacy download en_core_web_sm
```

## Run
To train and test the EA$^2$E model, use 
```sh
sh scripts/train_eaae.sh
sh scripts/test_bart_gen.sh
```
In Tesla P100 with 16 GB, the training takes about 15min for each epoch.

## Datasets
- ACE05 (Access from LDC[https://catalog.ldc.upenn.edu/LDC2006T06] and preprocessing following OneIE[http://blender.cs.illinois.edu/software/oneie/])
- WIKIEVENTS (contained in this repo, we use src/genie/no_ontology.py to remove the events with event types that are not in the ontology.)


## References
```
@inproceedings{EAAE,
  author    = {Qi Zeng and
               Qiusi Zhan and
               Heng Ji},
  title     = {EA$^2$E: Improving Consistency with Event Awareness for Document-level Argument Extraction},
  booktitle = {Proceedings of the 2022 Conference of the North American Chapter of
               the Association for Computational Linguistics: Human Language Technologies,
               {NAACL-HLT} 2022 Findings, Seattle, Washington, July 10-15, 2022},
  publisher = {Association for Computational Linguistics},
  year      = {2022}
}
```