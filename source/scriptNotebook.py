# %% [markdown]
# ## Bank Customer Segmentation
# <h2 style="font-weight:bold; font-family:sans-serif"><b>Goal of creating this Notebook</h2>
# 
# 1. Perform Clustering / Segmentation on the dataset and identify popular customer groups along with their definitions/rules
# 2. Perform Location-wise analysis to identify regional trends in India
# 3. Perform transaction-related analysis to identify interesting trends that can be used by a bank to improve / optimi their user experiences
# 4. Customer Recency, Frequency, Monetary analysis

# %% [markdown]
# **Table of contents of this notebook:**
# 
# **1.** Importing Necessary Libraries<br>
# **2.** Data Collection<br>
# **3.** Data Cleaning<br>
# **4.** Exploratory Data Analysis

# %% [markdown]
# <h2  style="text-align: center; padding: 20px; font-weight:bold">1. Importing Libraries</h2>

# %%
import re
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
import seaborn as sns

# %% [markdown]
# <h2  style="text-align: center; padding: 20px; font-weight:bold">2. Data Collection</h2><a id = "2"></a>

# %%
def import_data():
    """
    Imports our relative path Dataset and in addition to that, renames the TransactionAmount (INR) column in TransactionAmount, 
    this in order to avoid that in the future the special characters of the original name give us problems.
    
    Returns:
        DataFrame: dfGet source data
    """

