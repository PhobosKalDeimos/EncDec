import parser_args
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import logging
from tqdm import tqdm
import evaluate
import dataset
from models.encdec_ad.model import EncDecAD

logging.basicConfig(level=logging.INFO, format='%(message)s')

args = parser_args.parse_arguments()   
logging.info(f"Arguments: {vars(args)}")      

X_train = dataset.read_folder_normal(args.dataset_folder, args.frequency)
X_train, pipeline = dataset.preprocess_data(X_train)
Dataloader_train, DataLoader_val = dataset.split_data(X_train, args.train_split)

df_collision, X_collisions, df_test = dataset.read_folder_collisions(args.dataset_folder, args.frequency)
X_collisions = dataset.preprocess_data(X_collisions, pipeline, train=False)

model= EncDecAD(input_size=X_train.shape[1], lstm_layers=args.lstm_layers, encoding_size=args.latent_size, anomaly_window_size = args.window_size,   epochs=args.epochs_num)

model.fit(X_train, "./checkpoints/model_checkpoint.pth")
model.save(f"./checkpoints/model_fitted_{args.frequency}.pth")

# model.execute(ts=X_collisions, data_output= f"./output/scores/f{args.frequency}_")
logging.info("Testing the model...")
evaluate.evaluation(model, pipeline)

