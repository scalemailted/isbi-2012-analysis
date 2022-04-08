# ----------------------------------------------------
#[ 1 ]
import pandas as pd

from utils import *
from model.data_loader import *
from model.unet_multiresunet import MultiResUnet

# ----------------------------------------------------
#[ 2 ]
# Paths
train_pth = 'data/membrane/train'
test_pth = 'data/membrane/test'
results_pth = 'results/'
pretrained_pth = 'pretrained/'

# ----------------------------------------------------
# Experiment B – Slow test
# Train all models under test for 20 epochs of 250 training steps.
# ----------------------------------------------------
#[ 5 ]
# Half available image size to reduce training time
img_sz = (256, 256, 1)
batch_sz = 2

# Models
models = [
    ['unet_multiresunet', MultiResUnet,  dict(input_size=img_sz)]
]

# Data loaders
train_loader = loader(train_pth, input_generator_train, target_generator_train, batch_sz=batch_sz, img_sz=img_sz[:2])
test_loader = loader(test_pth, input_generator_test, target_generator_test, batch_sz=batch_sz, img_sz=img_sz[:2])

# ----------------------------------------------------
#[ 6 ]
# Results csv saved with this filename
test_title = '256px_250steps_20epochs'

# Experiment configuration
training_params = dict(
    train_steps=250, 
    val_steps=100, 
    epochs=20, 
    iterations=5,
    lr=1e-4
)

# Train models and record results
for model in models:
    print(f'\nTESTING MODEL: {model[0]}')
    save_pth = f'{pretrained_pth}{model[0]}_{test_title}.h5'
    results = test_model(model[1], train_loader, test_loader, **training_params, 
                         model_params=model[2], save_pth=save_pth)
    results_df = hists2df(results)
    results_df.to_csv(f'{results_pth}{model[0]}_{test_title}.csv')

# ----------------------------------------------------
# Experiment C – Fast test
# Train all models under test for 50 training steps. 
# Repeat each experiment for 20 iterations.
# ----------------------------------------------------
#[ 7 ]
# Results csv saved with this filename
test_title = '256px_50steps'

# Experiment configuration
training_params = dict(
    train_steps=50, 
    val_steps=100, 
    epochs=1, 
    iterations=20,
    lr=1e-4
)

# Train models and record results
for model in models:
    print(f'\nTESTING MODEL: {model[0]}')
    results = test_model(model[1], train_loader, test_loader, **training_params, 
                         model_params=model[2])
    results_df = hists2df(results)
    results_df.to_csv(f'{results_pth}{model[0]}_{test_title}.csv')

# ----------------------------------------------------
# Parameter count
# Count parameters in model
# ----------------------------------------------------
#[ 8 ]
# Count trainable parameters for each model
params = []
for model in models:
    m = model[1](**model[2])
    params.append([model[0], m.count_params()])

# ----------------------------------------------------
#[ 9 ]
pd.DataFrame(params, columns=['model', 'parameters'])

# ----------------------------------------------------
