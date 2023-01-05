# Story2Personality #

The dataset is a new narrative understanding benchmark to predict personality according to the character’s narrative texts in the script. We release the dataset and the codes for our work accepted to NAACL Student Research Workshop 2022: *[Machine Narrative Comprehension in Fictional Characters Personality Prediction Task](https://arxiv.org/)* and EMNLP 2022 *[MBTI Personality Prediction for Fictional Characters Using Movie Scripts](https://arxiv.org/)*.


## Step 0: Env Setup ##
``` bash
conda env create -f person_environment.yml python=3.8 pandas=1.5.2
conda activate person
python -m spacy download en
```

## Step 1: Data Parsing ## 
Our data parser first reads the narrative books and movie scripts from HTML files, and then extracts utterances said by recognized characters. The whole process can take 3~5 hours to finish. If you are **only interested in the data**, you can download them via [this link](https://1drv.ms/u/s!ArPzysVAJSvtquRqYKCnbUlg-Vst5A?e=fIKIVO) and unzip to the root folder.
``` bash
# move the downloaded "dialog_scene_mention_dicts.zip" to the root folder
unzip dialog_scene_mention_dicts.zip
```

If you would like to know how the raw text data is processed, you will have to download the HTML files first from [OneDrive](https://1drv.ms/u/s!ArPzysVAJSvtquRp-2lEfVB3FatIiA?e=Hin01I). The contents are the union of [NarrativeQA dataset](https://github.com/deepmind/narrativeqa) and [Movie-Script-Database](https://github.com/Aveek-Saha/Movie-Script-Database). Please unzip the downloaded file to the root repo folder. 
``` bash
# move the downloaded "raw_texts.zip" to the root folder
unzip raw_texts.zip
```
We are also sharing some other preprocessed files in the **preprocessed/** folder which are also the dependencies of our parser. The following command would generate *dialog_dict.pickle*, *scene_dict.pickle*, and *mention_dict.pickle* from scratch. 
```
python parse.py
```

Hereto, you will get three `.pickle` files which contain dictionaries of "what people say" and "who are mentioned" in a dialogue or a scene.


## Step 2: Model Training and Inferencing ##
To use the data for modeling, please go to [dataset/](dataset/) and download one of the tokenized datasets. The format is more readily for training and testing than those `.pickle` files. More details will be provided in the future.







# Citation #
If you find this repo useful, please consider citing our paper:
```bibtex
@article{sang2022mbti,
  title={MBTI Personality Prediction for Fictional Characters Using Movie Scripts},
  author={Sang, Yisi and Mou, Xiangyang and Yu, Mo and Wang, Dakuo and Li, Jing and Stanton, Jeffrey},
  journal={arXiv preprint arXiv:2210.10994},
  year={2022}
}
```