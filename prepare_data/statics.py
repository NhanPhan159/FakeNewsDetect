from data_clean import DataClean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

statics = DataClean(path='../dataset/preprocessing/test_vn.csv')
statics.readDataFromPath()
statics.dataFrameToArray()
statics.staticsData()
# statics.getFreq()
