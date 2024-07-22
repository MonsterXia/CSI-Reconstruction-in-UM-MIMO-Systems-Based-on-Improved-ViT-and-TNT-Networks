# -*- coding: utf-8 -*-
# @Author  : Monster_Xia
# @Time    : 2023/6/5 17:19
# @Function:

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib import rcParams

# Increase font size
rcParams.update({'font.size': 12})

# Define the data
compression_ratios = [1/4, 1/16, 1/32, 1/64, 1/128]
csi = [5.41, 3.84, 3.58, 3.45, None]
csa = [45.83, 44.26, 43.99, 43.86, None]
csi_vit = [1430.65, 241.27, 110.93, 53.04, 25.92]
csi_tnt = [3851.68, 695.53, 325.48, 157.17, 77.19]

# Multiply the values by 1,000,000 (1M=1,000,000)
csi = [x*1000000 if x is not None else x for x in csi]
csa = [x*1000000 if x is not None else x for x in csa]
csi_vit = [x*1000000 if x is not None else x for x in csi_vit]
csi_tnt = [x*1000000 if x is not None else x for x in csi_tnt]

# Create the chart
fig, ax = plt.subplots()
ax.plot(compression_ratios, csi, '-o', label='CSI-Net')
ax.plot(compression_ratios, csa, '-s', label='CSA-Net')
ax.plot(compression_ratios, csi_vit, '-^', label='CSI-ViT')
ax.plot(compression_ratios, csi_tnt, '-d', label='CSI-TNT')

# Add labels and legend
ax.set_xlabel('Compression Ratio')
ax.set_ylabel('FLOPs')
ax.legend(loc='upper left', bbox_to_anchor=(0.03, 0.97))
# ax.legend(loc='upper right')

# Set the formatter for the y-axis to use scientific notation
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-2, 3))
ax.yaxis.set_major_formatter(formatter)

# ax.legend()

# Add grid lines
ax.grid(True)

# Set the origin of the plot to the intersection of the x and y axes
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

# Show the chart
plt.savefig('../result/output/FLOPs.png')

plt.show()