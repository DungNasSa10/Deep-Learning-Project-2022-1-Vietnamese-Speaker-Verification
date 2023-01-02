# **Deep-Learning-Project-2022-1**

- [**Deep-Learning-Project-2022-1**](#deep-learning-project-2022-1)
  - [**Installation**](#installation)
  - [**Data and Output Folder**](#data-and-output-folder)
  - [**Datasets**](#datasets)
  - [**Crawling process**](#crawling-process)
  - [**Training**](#training)
  - [**Evaluation**](#evaluation)
  - [**Testing**](#testing)
  - [**Deployment**](#deployment)


## **Installation**
- Python version == 3.8
```
conda create -n dlds python=3.8 -y
conda activate dlds
pip install -r requirements.txt
```
- You also need to install FFmpeg. It can be downloaded through this [website](https://ffmpeg.org/download.html) or you can install it on Linux with apt with the following command: 
```
$ sudo apt update
$ sudo apt install ffmpeg
```

## **Data and Output Folder**

```
data
|
|───metadata
|   |
│   │───crawl
|   |       Voice list - An.csv
|   |       Voice list - Dung.csv
|   |       ...
|   |    
│   │───test
|   |   |
|   |   |───labels
|   |   |       private_t1_test_pairs_with_label.txt
|   |   |       private_t2_test_pairs_with_label.txt
|   |   |       public_test_pairs_with_label.txt
|   |   |
|   |   |───test_pairs
|   |           private_t1_test_pairs.txt 
|   |           private_t2_test_pairs.txt
│   │           public_test_pairs.txt
|   |   
│   |───train
|          training_metadata.txt
|
|───musan_augment
|
|───rir_noises
|
|───test
|   |
|   |───sv_vlsp_2021
|       |
|       |───private_test
|       |   |
|       |   |───competition_private_test
|       |
|       |───public_test
|           |
|           |───competition_public_test
|   
|───train
|   |
|   |───speaker_0
|   |   |
|   |   |───video_0
|   |   |       0.wav
|   |   |       1.wav
|   |   |       ...
|   |   |
|   |   |───video_1
|   |   |       0.wav
|   |   |       1.wav
|   |           ...
|   |
|   |───speaker_1
|   |   |
|   |   ...
|
|
output
|
|───checkpoints
|       ECAPA_CNN_TDNN.model
|       ECAPA_TDNN.model
|       RawNet3.model
|       SEResNet34.model
|       VGG_M_40.model

```

## **Datasets**
- You can find our datasets in Kaggle through this link [vietnamese-sv-datasets](https://www.kaggle.com/datasets/dungnasa10/vietnamese-speaker-verification). Note that in the this link, the structure of the dataset is a little different from the above folder structure but you don't need to worry about it. Only thing you need to do is change the file path properly or simply, just run our notebooks that we have provided for you.

## **Crawling process**
- We have prepared some csv files used for crawling in folder ```data/metadata/crawl```. You can create new files by make a copy of the template in this link [crawling-template](https://docs.google.com/spreadsheets/d/1z6By1Umim0xpomV0HyC4wG16B0KSKC2eGPobDzPrCbg/edit?usp=sharing), download  
in csv form and put it in the folder ```data/metadata/crawl```. 
- The following script will download all the videos in the csv file ```data/metadata/crawl/Voice list - An.csv```, convert them to .wav file, resample them into 16000 sample rate, use Silero VAD model to collect speech chunks, remove noisy audios and save the results in the folder ```data/train```. The argument -f refer to the path of the csv file, you can also put the link to the files saved in Google Drive while the argument -d specify the folder in which crawled data will be stored in.
```
python src/crawl.py -f https://drive.google.com/file/d/1BL4tkLkPPDeuYwlCaMvIrKYCst8E4zEN/view?usp=share_link -d ./data/train -sr 16000
```

## **Training**
We have prepared some config files for you. You can change the arguments in configuration file or pass individual arguments that are defined in trainSpeakerNet.py by --{ARG_NAME} {VALUE}. Note that the configuration file overrides the arguments passed via command line.

- Train RawNet3 with AAM-Softmax loss (the model checkpoints and validation results will be stored in folder ```output/RawNet3_AAM/```)
```
python src/learn.py --config src/learning/configs/VGG_M_40_AAM.yaml --train
```
- Train SE-ResNet34 with AAM-Softmax loss
```
python src/learn.py --config src/learning/configs/SEResNet34_AAM.yaml --train
```
- Train ECAPA-TDNN with Angular Prototypical loss
```
python src/learn.py --config src/learning/configs/ECAPA_TDNN_AP.yaml --train
```
- If you want to train on Kaggle, make a copy and run the Training part in this notebook [Vietnamese_SV](https://www.kaggle.com/code/dungnasa10/train-sv?scriptVersionId=115057715)

## **Evaluation**
- This command will eval the trained model SEResNet34 with Angular Prototypical loss on Public test and output the EER(%).
```
python src/learn.py --config src/learning/configs/SEResNet34_AP.yaml --eval
```
- This command will eval the trained model ECAPA CNN-TDNN with AAM-Softmax loss on T01 Private test.
```
python src/learn.py --config src/learning/configs/ECAPA_CNN_TDNN_AAN.yaml --eval
```
- This command will eval the trained model RawNet3 with AAM-Softmax loss on T02 Private test.
```
python src/learn.py --config src/learning/configs/RawNet3_AAN.yaml --eval
```
- Again, you can change the path of eval file in the config file. These eval files are saved in folder ```data/metadata/test/labels```
- If you want to train on Kaggle, make a copy and run the Evaluation part in this notebook [Vietnamese_SV](https://www.kaggle.com/code/dungnasa10/train-sv?scriptVersionId=115057715)

## **Testing**
- This command will test the trained model SEResNet34 with Angular Prototypical loss on Public test. The output is a csv file of the form ```audio_1 audio_2 similarity_score``` and be stored in ```output/testing_results/public_test```
```
python src/learn.py --config src/learning/configs/VGG_M_40_AP.yaml --test
```
- This command will eval the trained model ECAPA CNN-TDNN with AAM-Softmax loss on T01 Private test.
```
python src/learn.py --config src/learning/configs/ECAPA_CNN_TDNN_AAN.yaml --test
```
- This command will eval the trained model RawNet3 with AAM-Softmax loss on T02 Private test.
```
python src/learn.py --config src/learning/configs/RawNet3_AAN.yaml --test
```
- Again, you can change the path of test file in the config file. These test files are saved in folder ```data/metadata/test/test_pairs```
- If you want to train on Kaggle, make a copy and run the Testing part in this notebook [Vietnamese_SV](https://www.kaggle.com/code/dungnasa10/train-sv?scriptVersionId=115057715)

## **Deployment**
- You can download model checkpoints in [checkpoints](https://drive.google.com/drive/folders/1NQD_znYfCGELMoqavFcM0scBurlDcxjh) and put it in the folder ```output```
- Run the following command to test our deployment
```
streamlit run src/deploy.py --server.address 127.0.0.1 --server.port 8008
```
