# ----------------------------------------------------
#[ 1 ]
import pandas as pd
import seaborn as sns
import PIL

from model.data_loader import *

# ----------------------------------------------------
# Data augmentation
#
# Image transforms for test and training data
# ----------------------------------------------------
#[ 2 ]
train_pth = 'data/membrane/train'
test_pth = 'data/membrane/test'

# ----------------------------------------------------
#[ 3 ]
show_augmentation('data/membrane/train/input/0.png', input_generator_train, n_rows=1)
show_augmentation('data/membrane/train/target/0.png', target_generator_train, n_rows=1)

# ----------------------------------------------------
#[ 4 ]
show_augmentation('data/membrane/test/input/0.png', input_generator_test, n_rows=1)
show_augmentation('data/membrane/test/target/0.png', target_generator_test, n_rows=1)

# ----------------------------------------------------
# Train loader with transforms
# ----------------------------------------------------
#[ 5 ]
train_loader = loader(train_pth, input_generator_train, target_generator_train)
# ----------------------------------------------------
#[ 6 ]
show_sample(train_loader)

# ----------------------------------------------------
# Test loader without transforms
#
# Note some preprocessing was required to force test targets to render correctly. 
# Some information in test masks may have been lost. 
# See preprocessing.py for details.
# ----------------------------------------------------
#[ 7 ]
test_loader = loader(test_pth, input_generator_test, target_generator_test)

# ----------------------------------------------------
#[ 8 ]
show_sample(test_loader)

# ----------------------------------------------------



