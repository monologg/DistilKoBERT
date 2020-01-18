import os
import gdown


if not os.path.exists('data'):
    os.mkdir('data')


url = 'https://drive.google.com/uc?id=1-jXAnrHcKzzFiFhri37YOXH2OjP7DwMg'
output = 'data/dump.kobert.pickle.zip'

gdown.download(url, output, quiet=False)
