import pandas as pd 
import matplotlib.pyplot as plt 

# Load the datasets 
latent_space_loss_df = pd.read_csv('latent_space_loss.csv', delimiter=',')
compression_analysis_df = pd.read_csv('compression_analysis.csv', delimiter=',')

plt.figure(figsize=(10, 6))

plt.plot(latent_space_loss_df['Latent Space Dimension (bytes)'], latent_space_loss_df['Reconstruction Error (MSE Loss)'], label='Latent Space Loss', marker='o', linestyle='-', color='b')
plt.scatter(compression_analysis_df['Bytes'], compression_analysis_df['MSE Error'], label='Compression Analysis', color='r')

plt.yscale('log')
plt.title('Comparison of Latent Space Loss and Compression Analysis')
plt.xlabel('Dimension / Bytes')
plt.ylabel('Log MSE Error')
plt.legend()
plt.grid(True)
plt.show()