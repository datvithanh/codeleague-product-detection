import numpy as np
import pandas as pd
import torch
from torchvision import transforms, models
from tqdm import tqdm 

from dataset import LoadDataset
from utils import load_image, Normaliztion


device = torch.device('cuda:2') if torch.cuda.is_available() else torch.device('cpu') 

#device = torch.device('cpu')

#model_path = 'ckpt/22-06/model_epoch17'
#model_path = 'ckpt/22-06-aug/model_epoch24_0.777'
#model_path = 'ckpt/22-06-enet/model_epoch4_0.7589'
#model_path = 'ckpt/22-06/model_epoch8_0.7619'
model_path = 'ckpt/03-07/model_epoch7_0.7631'
model_path = 'ckpt/03-07-lr/model_epoch3_0.8108000000000001'

if device.type == 'cpu':
    model = torch.load(model_path, map_location='cpu')
else:
    model = torch.load(model_path).to(device)

transform = transforms.Compose([Normaliztion()])

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
    X = load_image(X)
    X = Normaliztion(X)
    X = torch.Tensor(np.array([X]))
    preds, scores = predict(X)
    return preds, scores

def infer(batch_paths):
    X = [transform(load_image(tmp)).copy() for tmp in batch_paths]
    X = torch.Tensor(np.array(X)).to(device)
    preds, scores = predict(X)
    return preds, scores

if __name__ == "__main__":
    data_path = 'csv/test.csv'
    df = pd.read_csv(data_path)
    nparr = np.array(df)
    np.random.shuffle(nparr)

    batch, batch_label = [], []
    total, cnt = 0, 0
    filename = data_path.split('/')[-1][:-4]

    with open(f'{filename}_prediction.txt', 'w+') as f:
        for path, label in tqdm(zip(df['image'], df['label']), total=len(df['image'])):
            cnt += 1
            batch.append(path.replace('/home/datvt/hust/shopee-codeleague/test/test', '/home2/htthanh/DatVT/code-league/test'))
            batch_label.append(int(label))
            if len(batch) == 10 or cnt == len(df['image']):
                preds, scores = infer(batch)
                for path, pred, label, score in zip(batch, preds, batch_label, scores):
                    f.write(f'{path} {pred} {label} {list(score)}\n')

                total += sum([tmp1 == tmp2 for tmp1, tmp2 in zip(preds, batch_label)])

                batch, batch_label = [], []
            # if cnt % 1000 == 0:
            #     print(total)

    # print(total, len(nparr))


