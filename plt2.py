import matplotlib.pyplot as plt
import csv
import numpy as np

from pylab import *
from matplotlib.ticker import  MultipleLocator
from matplotlib.ticker import  FormatStrFormatter

#  将x主刻度标签设置为20的倍数(也即以 20为主刻度单位其余可类推)
xmajorLocator = MultipleLocator(10);

# 设置x轴标签文本的格式
xmajorFormatter = FormatStrFormatter('%d') 

# 将x轴次刻度标签设置为5的倍数
xminorLocator = MultipleLocator(5) 

# 设定y 轴的主刻度间隔及相应的刻度间隔显示格式

# 将y轴主刻度标签设置为1.0的倍数
ymajorLocator = MultipleLocator(0.005)

#  设置y轴标签文本的格式
ymajorFormatter = FormatStrFormatter('%1.3f')

# 将此y轴次刻度标签设置为0.2的倍数
yminorLocator = MultipleLocator(0.001)

x = []
y_precision_w = []
y_recall_w = []
y_f1_w = []
y_precision_k = []
y_recall_k = []
y_f1_k = []

with open('./result/www-topics.csv') as file:
    www_data = file.readlines()
for row in www_data:
    [topic, precision, recall, f1_score] = row.split(',')
    x.append(int(topic))
    y_precision_w.append(float(precision))
    y_recall_w.append(float(recall))
    y_f1_w.append(float(f1_score))

with open('./result/kdd-topics.csv') as file:
    kdd_data = file.readlines()
for row in kdd_data:
    [topic, precision, recall, f1_score] = row.split(',')
    y_precision_k.append(float(precision))
    y_recall_k.append(float(recall))
    y_f1_k.append(float(f1_score))


www_plot = plt.subplot(211)
plt.title('WWW')
plt.plot(x, y_precision_w, '-bo', label='Precision')
plt.plot(x, y_recall_w, '-rD' ,label='Recall')
plt.plot(x, y_f1_w, '-gs', label='F1-score')
ax = plt.gca()
ax.axis([-1, 101, 0.137, 0.151]) 
ax.set_xlabel('$\mathit{K}$')
www_plot.legend(loc='higher right')
#设置主刻度标签的位置,标签文本的格式
www_plot.xaxis.set_major_locator(xmajorLocator)
www_plot.xaxis.set_major_formatter(xmajorFormatter)

www_plot.yaxis.set_major_locator(ymajorLocator)
www_plot.yaxis.set_major_formatter(ymajorFormatter)

#显示次刻度标签的位置,没有标签文本
www_plot.xaxis.set_minor_locator(xminorLocator) 
www_plot.yaxis.set_minor_locator(yminorLocator)

www_plot.xaxis.grid(False, which='major') #x坐标轴的网格使用主刻度
www_plot.yaxis.grid(False, which='minor') #y坐标轴的网格使用次刻度

kdd_plot = plt.subplot(212)
plt.title('KDD')
plt.plot(x, y_precision_k, '-bo', label='Precision')
plt.plot(x, y_recall_k, '-rD' ,label='Recall')
plt.plot(x, y_f1_k, '-gs', label='F1-score')
ax = plt.gca()
ax.axis([-1, 101, 0.151, 0.162]) 
ax.set_xlabel('$\mathit{K}$')
kdd_plot.legend(loc='higher right')
#设置主刻度标签的位置,标签文本的格式
kdd_plot.xaxis.set_major_locator(xmajorLocator)
kdd_plot.xaxis.set_major_formatter(xmajorFormatter)

kdd_plot.yaxis.set_major_locator(ymajorLocator)
kdd_plot.yaxis.set_major_formatter(ymajorFormatter)

#显示次刻度标签的位置,没有标签文本
kdd_plot.xaxis.set_minor_locator(xminorLocator)
kdd_plot.yaxis.set_minor_locator(yminorLocator)

kdd_plot.xaxis.grid(False, which='major') #x坐标轴的网格使用主刻度
kdd_plot.yaxis.grid(False, which='minor') #y坐标轴的网格使用次刻度

plt.show()