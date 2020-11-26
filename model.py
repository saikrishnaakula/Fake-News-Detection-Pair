import config
import dataset
import torch
import time
import transformers
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import os

class BertForFakeNewsDetection:

    def __init__(self):
        self.model = config.MODEL
        self.device = torch.device("cpu")
    
    def get_features(self,train=True):

        fakenews_dataset = dataset.FakeNewsDataset(train=train)
        n_samples = fakenews_dataset.n_samples
        dataloader = DataLoader(dataset=fakenews_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
        dataiter = iter(dataloader)
        if train == True:
            labels = []
        features = []
        if config.READ ==False:
            with torch.no_grad():
                for ind, d in tqdm(enumerate(dataloader),total=len(dataloader)):
                    if train == True:
                        label = np.array(d['labels'])
                        labels.extend(label)
                    input_ids = torch.tensor(d['input_ids']).to(self.device).long()
                    attention_mask = d['attention_mask']
                    segment_ids = torch.tensor(d['segment_ids']).to(self.device).long()
                    outputs = self.model(input_ids,token_type_ids = segment_ids, attention_mask=attention_mask)
                    last_hidden_state = outputs[1]
                    features.append(last_hidden_state[-1][:,0,:].numpy())
                    if (ind+1)%1000 == 0:
                        np.save(f"Saved_Model/train/data_{ind+1}.npy",np.array(features))
                        features = []
            np.save(f"Saved_Model/train/data_last.npy",np.array(features))
            features = []
            files = os.listdir('Saved_Model/train/')
            for f in files:
                for a in np.load(os.path.join("Saved_Model/train", f)):
                    features.append(a)
            features = np.array(features).reshape(n_samples,768)
        else:
            if train==True:
                features = np.load("Saved_Model/data.npy")
            else:
                features = np.load("Saved_Model/train.npy")

            features = np.array(features)
            # .reshape(n_samples,768) try this as well

        if train == True:
            if config.READ == False:
                labels = np.array(labels)
            else:
               labels = np.load("Saved_Model/labels.npy")
            return features, labels
        else:
            return features