import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the dataset
data = pd.read_csv('TumorVariantDownload_r21-2.csv')

#count and calculate percentage
grouped_topography = data['Sub_topography'].value_counts(ascending=True)/len(data)*100
# select types that occur in more than 0.9% of cases
f = grouped_topography[grouped_topography>0.9]

# we can use map() and lambda functions to create text for
# bar labels to be shown
bar_labels = list(map(lambda x: f"{x:.2f} %", f))

plt.figure(figsize=(7,8))
barplot = plt.barh(np.arange(0,len(f)), f, color='thistle')
# plot bar labels at the edges of bars inside the figure
plt.bar_label(barplot, labels=bar_labels, label_type='edge', padding=2)
# increase the ax. limit to make enough space for bar labels
plt.xlim([0, f.iloc[-1] + 2])
# write cancer type on y axes
plt.yticks(ticks=np.arange(0,len(f)), labels=f.keys(), fontsize=9)#, rotation=75)
plt.ylabel('cancer type', fontsize = 15)
plt.xlabel('occurence [%]', fontsize = 12)
plt.title('Cancer type occurence in the dataset', fontsize = 12)
plt.tight_layout()
plt.savefig('cancer_occurrence_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# count
grouped = data['Effect'].value_counts()
# calculate percentage, two decimals
percentage = grouped / sum(grouped) * 100

# group mutation with corresponding percentage to be used for labels
# Note: I swapped grouped.keys() for grouped.index, which is standard Pandas syntax
l = list(map(lambda X: X[0] + f' - {X[1]:.2f} %', zip(grouped.index, percentage)))

# FIX: Plot directly from the Series, removing pd.DataFrame() and y='Effect'
plt.figure(figsize=(6,6))
p = grouped.plot.pie(labeldistance=None, labels=l, fontsize=6)

p.set_title('Mutation Effect', fontsize=12, fontweight='bold')
p.set_ylabel('') # This removes the default, often messy y-axis label Pandas adds
plt.tight_layout()
plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('mutation_effect_pie.png', dpi=300, bbox_inches='tight')

plt.show()


codon_number = data['Codon_number']

# we create bins in range from min to max of codon number
hist, bins = np.histogram(codon_number, bins = np.arange(min(codon_number), max(codon_number) + 1, 1))
# let's center the bins to adapt output to the real scenario
center = (bins[:-1] + bins[1:]) / 2

plt.figure(figsize=(10,5))
# plot probabilities
plt.bar(center[1:], hist[1:]/len(codon_number)*100, width=1, color = '#957dad')
plt.ylabel('percentage [%]', fontsize=15)
plt.xlabel('codon number', fontsize=15)
plt.title('Codon number distribution (only exons shown)')
plt.xticks(np.arange(0, max(codon_number)+10, 20))

plt.text(0, 6, f'Exon mutations (shown here) take {100-hist[0]/len(codon_number)*100:.2f} % of total mutations,'
         , fontsize=10)
plt.text(0, 5.5, f'while remaining {hist[0]/len(codon_number)*100:.2f} % are intron mutations'
         , fontsize=10)
plt.savefig('codon_number_distribution.png', dpi=300, bbox_inches='tight')
plt.show()