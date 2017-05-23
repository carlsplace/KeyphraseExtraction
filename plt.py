import matplotlib.pyplot as plt
import csv
import numpy as np

from pylab import *
from matplotlib.ticker import  MultipleLocator
from matplotlib.ticker import  FormatStrFormatter

#  将x主刻度标签设置为20的倍数(也即以 20为主刻度单位其余可类推)
xmajorLocator = MultipleLocator(2);

# 设置x轴标签文本的格式
xmajorFormatter = FormatStrFormatter('%d') 

# 将x轴次刻度标签设置为5的倍数
xminorLocator = MultipleLocator(5) 

# 设定y 轴的主刻度间隔及相应的刻度间隔显示格式

# 将y轴主刻度标签设置为1.0的倍数
ymajorLocator = MultipleLocator(0.01)

#  设置y轴标签文本的格式
ymajorFormatter = FormatStrFormatter('%1.2f')

# 将此y轴次刻度标签设置为0.2的倍数
yminorLocator = MultipleLocator(0.005)

x = [2, 3, 5, 10]
y_precision_w = [0.148048452, 0.142857143, 0.144280968, 0.140009492]
y_recall_w = [0.149659864, 0.145200193, 0.146647371, 0.142305837]
y_f1_w = [0.148849797, 0.144019139, 0.145454545, 0.141148325]
y_precision_k = [0.160669456,0.151890034,0.145517241,0.142857143]
y_recall_k = [0.16,0.150237933,0.143439837,0.140720598]
y_f1_k = [0.160334029,0.151059467,0.144471072,0.141780822]

www_plot = plt.subplot(121)
plt.title('WWW')
plt.plot(x, y_precision_w, '-bo', label='Precision')
plt.plot(x, y_recall_w, '-rD' ,label='Recall')
plt.plot(x, y_f1_w, '-gs', label='F1-score')
ax = plt.gca()
ax.axis([1.5, 10.5, 0.14, 0.15]) 
ax.set_xlabel('window')
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

kdd_plot = plt.subplot(122)
plt.title('KDD')
plt.plot(x, y_precision_k, '-bo', label='Precision')
plt.plot(x, y_recall_k, '-rD' ,label='Recall')
plt.plot(x, y_f1_k, '-gs', label='F1-score')
ax = plt.gca()
ax.axis([1.5, 10.5, 0.14, 0.161]) 
ax.set_xlabel('window')
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