import pandas as pd
import csv
from collections import Counter
import itertools
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import pdb
from tqdm import tqdm
import argparse
import datetime as dt
import pickle
import unicodedata
import re
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
import string
###############################
from IPython import embed
###############################
USE_BOW = False
USE_CUDA = torch.cuda.is_available()
PAD_token = 0
UNK_token = 1