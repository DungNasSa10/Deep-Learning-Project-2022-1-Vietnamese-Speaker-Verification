# Deep-Learning-Project-2022-1

## Installation 
```
conda create -n dlds python=3.8 -y
conda activate dlds
pip install -r requirements.txt
```

## Crawling
+ Change path for urls file
```
python src/crawl.py -f https://drive.google.com/file/d/1BL4tkLkPPDeuYwlCaMvIrKYCst8E4zEN/view?usp=share_link -d ./data/train 
```

## Training
```
python src/trainer.py --train --config src/learning/configs/SEResNet34_AP.yaml
```

## Evaluation
```
python src/trainer.py --eval --config src/learning/configs/SEResNet34_AP.yaml --eval
```

## Testing
```
python src/trainer.py --test --config src/learning/configs/SEResNet34_AP.yaml --test
```
