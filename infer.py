import numpy as np
import pandas as pd
import torch
from torchvision import models
from tqdm import tqdm 

from dataset import LoadDataset
from utils import load_image


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 

if device.type == 'cpu':
    model = torch.load('ckpt/mask-hsv/model_epoch0',map_location='cpu')
else:
    model_path = 'ckpt/22-06/model_epoch17'
    # model = torch.load('ckpt/movement/model_epoch4')
    # model = torch.load('ckpt/no-movement/model_epoch5')
    model = torch.load(model_path)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)[:, None]

def predict(X):
    X = X.to(device = device,dtype=torch.float32)
    outputs = model(X)
    preds = torch.max(outputs, 1)[1]
    return preds.tolist(), softmax(np.array(outputs.tolist()))

def predict_image(X):
    X = load_image(X, path=False)
    X = torch.Tensor(np.array([X]))
    print(X.shape)
    preds, scores = predict(X)
    return preds, scores

def infer(batch_paths):
    X = [load_image(tmp) for tmp in batch_paths]
    X = torch.Tensor(np.array(X))
    preds, scores = predict(X)
    return preds, scores

if __name__ == "__main__":
    data_path = 'csv/22-06/val.csv'
    df = pd.read_csv(data_path)
    nparr = np.array(df)
    np.random.shuffle(nparr)

    batch, batch_label = [], []
    total, cnt = 0, 0
    filename = data_path.split('/')[-1][:-4]

    with open(f'{filename}_prediction.txt', 'w+') as f:
        for path, label in tqdm(zip(df['image'], df['label']), total=len(df['image'])):
            cnt += 1
            batch.append(path)
            batch_label.append(0 if label == 'real' else 1)
            if len(batch) == 16:
                preds, scores = infer(batch)
                for path, pred, label, score in zip(batch, preds, batch_label, scores):
                    f.write(f'{path} {pred} {label} {score}\n')

                total += sum([tmp1 == tmp2 for tmp1, tmp2 in zip(preds, batch_label)])

                batch, batch_label = [], []
            # if cnt % 1000 == 0:
            #     print(total)

    # print(total, len(nparr))


