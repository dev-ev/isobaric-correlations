#!/usr/bin/env python
# coding: utf-8

# <h2>Multiple faces of correlation in isobaric labeling-based proteomics</h2>

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set(font_scale = 1.2)
sns.set_style('whitegrid')
sns.set_palette('muted')


# In[2]:


def rename_abund_columns(dfIn):
    '''Adapted for Peptide Groups table'''
    renaming_dict = {}
    for c in dfIn.columns:
        if 'Abundances Normalized' in c:
            new_name = c.split('Sample ')[-1]
            new_name = new_name.split('_')[0]
            renaming_dict[c] = new_name
    dfOut = dfIn.rename(renaming_dict,axis='columns')
    return dfOut


# <h3>Import and clean up the data</h3>

# The isobaric labeling LC-MS data was processed using the vendor software that produces a bunch of tab-delimited text files. [The LC-MS data is available](https://www.ebi.ac.uk/pride/archive/projects/PXD005236) through the ProteomeXChange repository. You can find the Proteins file in [the Github repo](https://github.com/dev-ev/prot-rna-umap-clustering/tree/main/Prot_Data).

# In[3]:


df = pd.read_csv(
    os.getcwd() + '/Prot_Data/Ecoli_Isobaric_Abundances_Proteins.txt',
    sep='\t'
)
print(df.shape)
df = rename_abund_columns(df)
print(df.columns)
df.head(3)


# In[4]:


df['Protein FDR Confidence Combined'].unique()


# In[5]:


df['Master'].unique()


# In[6]:


df[
    df['Accession'].str.contains('cont')
].shape


# Filter out the contaminant proteins that don't belong to <i>E coli</i>

# In[7]:


df = df[ ~df['Accession'].str.contains('cont') ]
print(df.shape)
df.head(3)


# Uniprot Accessions are unique identifiers in this data set, why not use them as row indices

# In[8]:


df.set_index('Accession', inplace = True)
df.head(3)


# Select the numeric values

# In[9]:


dfNum = df[
    ['S01', 'S02', 'S03', 'S04', 'S05',
     'S06', 'S07', 'S08', 'S09', 'S10']
].dropna(axis = 'rows').copy()
print(dfNum.shape)
dfLog = np.log10( dfNum )
dfLog.head(3)


# In[10]:


dfLog.tail(3)


# In[11]:


#dfLog.hist(color = '#4a966b', bins = 20,
#           layout = (2, 5), figsize = (15, 5),
#           sharex = True)
dfLog.hist(color = '#4f6fd2', bins = 20,
           layout = (2, 5), figsize = (15, 5),
           sharex = True)
plt.suptitle('Log10 Abundances')


# In[12]:


c = '#4f6fd2'
dfLog.boxplot(
    figsize = (8, 5),
    notch = True, showmeans = False, vert = True,
    showfliers = False,
    boxprops = dict(linewidth = 2, color = c),
    whiskerprops = dict(linewidth = 1, color = c),
    capprops = dict(linewidth = 1, color = c),
    medianprops= dict(linewidth = 2)
)
plt.title('Log10 Normalized Signal Abundances', fontsize = 16)
#plt.grid(b=None)


# In[27]:


dfNum.sum()


# In[28]:


dfLog.sum()


# In[13]:


dfLog.mean()


# <h3>Look at protein-protein correlations</h3>

# In[14]:


sns.clustermap(
    dfLog.T.corr(method='pearson'),
    figsize=(10,10),
    cmap='coolwarm', vmin=-1, vmax=1,
    annot=False, cbar=True, linewidth=0
)


# Select upper triangle of the correlaion matrix, exclude the diagonal and flatten the array.

# In[15]:


corrArray = dfLog.T.corr(method='pearson').to_numpy()
corrArray = corrArray[
    np.triu_indices( corrArray.shape[0], k = 1)
]
print( len(corrArray) )
corrArray


# In[16]:


f = plt.figure(figsize=(10,3))
plt.hist(corrArray, bins=100)
plt.title('Protein-Protein Correlations (Normalized Abundances)')


# In[17]:


pd.DataFrame(corrArray).describe()


# Create the data set with the averaged biological replicates, and see how the distribution of the correlations change

# In[20]:


def add_group_averages(df, groups):
    df = df.copy()
    for g in groups:
        group = np.array(df[groups[g]])
        averages = np.nanmean(group, axis=1)
        df[('AVG '+g)] = averages
    return df


# In[21]:


groups = {
    'Control': ['S01', 'S02'],
    'Condition 1': ['S03', 'S04'], 'Condition 2': ['S05', 'S06'],
    'Condition 3': ['S07', 'S08'], 'Condition 4': ['S09', 'S10']
}
dfAvg = add_group_averages(dfNum, groups)
dfAvg = np.log10(dfAvg)
dfAvg.head(3)


# Seems correct. Let's go ahead and see the disctribution of correlations on the average vals.

# In[26]:


sns.clustermap(
    dfAvg.iloc[: , -5:].T.corr(method='pearson'),
    figsize=(10,10),
    cmap='coolwarm', vmin=-1, vmax=1,
    annot=False, cbar=True, linewidth=0
)


# In[25]:


corrArray = dfAvg.iloc[: , -5:].T.corr(method='pearson').to_numpy()
corrArray = corrArray[
    np.triu_indices( corrArray.shape[0], k = 1)
]
print( len(corrArray) )
f = plt.figure(figsize=(10,3))
plt.hist(corrArray, bins=100, color = '#1b7c4c')
plt.title('Protein-Protein Correlations (Averaged Biological Conditions)')


# In[23]:


dfAvg.iloc[: , -5:].T


# Simulate the data set with improper normalization by changing the values for S02, S04 and S08

# In[28]:


10**0.2


# In[29]:


dfDistorted = dfLog.copy()
dfDistorted['S02'] = dfLog['S02'] + 0.2
dfDistorted['S04'] = dfLog['S04'] - 0.2
rArr = np.random.rand( len(dfLog['S08']) ) * 0.5 + 0.7
dfDistorted['S08'] = dfLog['S08'] * rArr
dfDistorted.head(3)


# In[30]:


dfDistorted.mean()


# In[31]:


dfDistorted.hist(color = '#6b4a96', bins = 20,
           layout = (2, 5), figsize = (15, 5), sharex = True)
plt.suptitle('Log10 Abundances (Poor Normalization)')


# In[33]:


c = '#6b4a96'
dfDistorted.boxplot(
    figsize = (8, 5),
    notch = True, showmeans = False, vert = True,
    showfliers = False,
    boxprops = dict(linewidth = 2, color = c),
    whiskerprops = dict(linewidth = 1, color = c),
    capprops = dict(linewidth = 1, color = c),
    medianprops= dict(linewidth = 2)
)
plt.title('Log10 Signal Abundances (Poor Normalization)', fontsize = 16)


# In[34]:


sns.clustermap(
    dfDistorted.T.corr(method='pearson'),
    figsize=(10,10),
    cmap='coolwarm', vmin=-1, vmax=1,
    annot=False, cbar=True, linewidth=0
)


# In[35]:


corrArray = dfDistorted.T.corr(method='pearson').to_numpy()
corrArray = corrArray[
    np.triu_indices( corrArray.shape[0], k = 1)
]
print( len(corrArray) )
f = plt.figure(figsize=(10,3))
plt.hist(corrArray, bins=100, color = '#6b4a96')
plt.title('Protein-Protein Correlations (Poor Normalization)')


# Scale each row on row mean

# In[36]:


dfScaled = dfNum.T.copy()
dfScaled = dfScaled / dfScaled.mean()
dfScaled = dfScaled.T
dfScaled.head(3)


# In[37]:


dfScaled.describe()


# In[38]:


dfScaled = np.log10(dfScaled)
dfScaled.head(3)


# In[50]:


dfScaled.hist(color = '#964a54', bins = 50,
           layout = (5, 2), figsize = (15, 8),
           sharex = True)
plt.suptitle('Log10 Scaled Abundances')


# In[56]:


c = '#964a54'
dfScaled.boxplot(
    figsize = (8, 5),
    notch = True, showmeans = False, vert = True,
    showfliers = False,
    boxprops = dict(linewidth = 2, color = c),
    whiskerprops = dict(linewidth = 1, color = c),
    capprops = dict(linewidth = 1, color = c),
    medianprops= dict(linewidth = 2)
)
plt.title('Log10 Scaled Abundances', fontsize = 16)


# In[57]:


corrArray = dfScaled.T.corr(method='pearson').to_numpy()
corrArray = corrArray[
    np.triu_indices( corrArray.shape[0], k = 1)
]
print( len(corrArray) )
f = plt.figure(figsize=(10,3))
plt.hist(corrArray, bins=100, color = '#964a54')
plt.title('Protein-Protein Correlations (Scaled Proteins)')


# <h3>Look at sample-sample correlations</h3>

# In[41]:


pd.plotting.scatter_matrix(
    dfLog, figsize = (11, 10), diagonal = 'kde'
)
plt.suptitle('Sample-Sample Correlations on Abundances')


# In[42]:


dfLog.corr(method='pearson')


# In[43]:


f = plt.figure(figsize=(8,6))
sns.heatmap(
    dfLog.corr(method='pearson').round(2),
    square=True, cmap='coolwarm', vmin=-1, vmax=1,
    annot=False, cbar=True, linewidth=4, linecolor='white'
)
plt.suptitle('Sample-Sample Correlations on Abundances',fontsize=16)


# Now let's check how the sample-sample correlations would look on the scaled values

# In[44]:


pd.plotting.scatter_matrix(
    dfScaled, figsize = (11, 10), diagonal = 'kde'
)
plt.suptitle('Sample-Sample Correlations on Scaled Abundances')


# In[45]:


f = plt.figure(figsize=(8,6))
sns.heatmap(
    dfScaled.corr(method='pearson').round(2),
    square=True, cmap='coolwarm', vmin=-1, vmax=1,
    annot=False, cbar=True, linewidth=4, linecolor='white'
)
plt.suptitle('Sample-Sample Correlations on Scaled Abundances',fontsize=14)


# Scale on sample S01

# In[63]:


dfScaledOnS1 = dfNum.T.copy()
dfScaledOnS1 = dfScaledOnS1 / dfScaledOnS1.loc['S01']
dfScaledOnS1 = dfScaledOnS1.T
dfScaledOnS1.head(3)


# In[72]:


dfScaledOnS1 = np.log10(dfScaledOnS1)
dfScaledOnS1.head(3)


# In[74]:


pd.plotting.scatter_matrix(
    dfScaledOnS1.iloc[:,1:],
    figsize = (11, 10), diagonal = 'kde'
)
plt.suptitle('Sample-Sample Correlations / Abundances Scaled on S01')


# In[80]:


f = plt.figure(figsize=(7.5,5.5))
sns.heatmap(
    dfScaledOnS1.iloc[:,1:].corr(method='pearson').round(2),
    square=True, cmap='coolwarm', vmin=-1, vmax=1,
    annot=False, cbar=True, linewidth=4, linecolor='white'
)
plt.suptitle('Sample-Sample Correlations / Abundances Scaled on S01',fontsize=14)

