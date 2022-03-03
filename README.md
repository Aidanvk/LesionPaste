# LesionPaste
This is the official pytorch implementation of the paper:
> ********** [[link](https://link.springer.com/chapter/)] [[arxiv](https://arxiv.org/)]
## Dataset

Three publicly-accessible datasets are used in this work.

- EyeQ: Images of grades 1-4 are all considered as abnormal. All normal images in the training set are used to train LesionPaste. [[link](https://github.com/HzFu/EyeQ)].
- IDRiD: The lesions of a single fundus image from IDRiD are used as the true anomalies for DR anomaly detection [[link](https://idrid.grand-challenge.org)].
- MosMed: CT slices containing COVID-19 lesions are considered as abnormal. [[link](https://www.kaggle.com/mathurinache/mosmeddata-chest-ct-scans-with-covid19)].


## Usage

### LesionPaste network

A trained model and predicted results can be downloaded [here](https://github.com/YijinHuang/Lesion-based-Contrastive-Learning/releases/tag/v1.0).


**1. Use one of the following method to build your dataset:**


Organize your images as follows:

```
├── your_data_dir
    ├── train
        ├── Normal
            ├── image1.jpg
            ├── image2.jpg
            ├── ...
        ├── Abnormal
            ├── image3.jpg
            ├── image4.jpg
            ├── ...
    ├── test
        ├── Normal
            ├── image5.jpg
            ├── image6.jpg
            ├── ...
        ├── Abnormal
            ├── image7.jpg
            ├── image8.jpg
            ├── ...
```
Then replace the value of 'data_path' in BASIC_CONFIG in `config.py` with path to your_data_dir.

Recommended environment:

- python 3.8+
- pytorch 1.5.1
- torchvision 0.6.1
- tensorboard 2.2.1
- tqdm

To install the dependencies, run:
```shell
$ git clone https://github.com/Aidanvk/LesionPaste.git
$ cd LesionPaste
$ pip install -r requirements.txt
```

**2. Update your training configurations and hyperparameters in `train.py`.**

**3. Run to train:**

```shell
$ CUDA_VISIBLE_DEVICES=x python main.py
```

