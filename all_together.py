import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

# Load the datasets 
jpeg = pd.read_csv('data/jpeg.csv')
auto = pd.read_csv('data/autoencoders.csv')
# rast = pd.read_csv('data/raster.csv')
rast = pd.read_csv('raster/32.csv')

plt.scatter(jpeg.iloc[:, 0], jpeg.iloc[:, 1], label="JPEG")
plt.scatter(2*auto.iloc[:, 0], auto.iloc[:, 1], label="Autoencoding")
plt.scatter(rast.iloc[:, 0], rast.iloc[:, 1], label="Raster")
plt.legend()
plt.title("Altogether")
plt.xlabel("Bytes")
plt.ylabel("Mean-Squared Error")
plt.yscale("log")
plt.show()