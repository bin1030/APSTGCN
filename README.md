# APSTGCN: Adaptive Parallel Spatio-Temporal Graph Convolution Network for Traffic Forecasting

This is PyTorch implementation of APSTGCN :

## Requirements

- scipy
- torch
- tqdm
- h5py
- numpy
- pandas
- PyYAML
- tensorboardX
- torch

Dependency can be installed using the following command:

```
pip install -r requirements.txt
```

## Data Preparation

We have included the METR-LA dataset in the code. Other datasets can be downloaded and used from [Google Drive](https://drive.google.com/drive/folders/18bQJ_1wI4YkeMh2ZxbiDEwRbm4QITTUP?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/1dvMKLyAk5fOXQtY0cED9Ig?pwd=6666).

Run the following commands to generate train/test/val dataset at `data/{METR-LA}/{train,val,test}.npz`.

```
# METR-LA
python -m utils.generate_training_data --output_dir=data/METR-LA --traffic_df_filename=data/METR-LA/metr-la.h5
```
## Model Training

We utilize the A  two-stage  training  strategy to further enhance the model  performance. Here are commands for training the model on `METR-LA`. 

```
#first stage
python run1.py --config metr-la-stage1-config --name stage-1
#seconde stage
python run2.py --config metr-la-stage2-config --name stage-2
```

Here are commands for testing the model on `METR-LA`. 

```
python run2.py --config metr-la-stage2-config --name stage-2 --test
```
Due to the needs of subsequent work, we are currently providing a simplified version of the model and example implementations. The full project code will be gradually released in future updates.

We appreciate your understanding. If you have any specific questions or requests regarding the code, please feel free to contact us.





