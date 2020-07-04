import pandas as pd
lines = open('test_prediction.txt', 'r').readlines()

lines = [(tmp.split()[0].split('/')[-1], int(tmp.split()[1])) for tmp in lines]
print(lines[:10])
fp, cat = zip(*lines)
df = pd.DataFrame.from_dict({'filename': fp, 'category': cat})
df.to_csv('submission.csv', index=False)
