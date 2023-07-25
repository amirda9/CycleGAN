import torch 
import pandas as pd
import torch.nn as nn
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn import preprocessing
from Arch import GeneratorState, GeneratorMeas, DiscriminatorState, DiscriminatorMeas
import datetime
from torch.utils.data import DataLoader,Dataset


G_meas2state = GeneratorState()
G_meas2state.load_state_dict(torch.load('./G_measurement2state_21-07-2023_19-40-53.pth', map_location=torch.device('cpu')))
G_state2meas = GeneratorMeas()
G_state2meas.load_state_dict(torch.load('./G_state2measurement_21-07-2023_19-40-53.pth', map_location=torch.device('cpu')))
D_state = DiscriminatorState()
D_state.load_state_dict(torch.load('./D_state_21-07-2023_19-40-53.pth', map_location=torch.device('cpu')))
D_meas = DiscriminatorMeas()
D_meas.load_state_dict(torch.load('./D_measurement_21-07-2023_19-40-53.pth', map_location=torch.device('cpu')))



