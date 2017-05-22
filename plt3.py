import matplotlib.pyplot as plt
import csv
import numpy as np

from pylab import *
from matplotlib.ticker import  MultipleLocator
from matplotlib.ticker import  FormatStrFormatter

#  将x主刻度标签设置为20的倍数(也即以 20为主刻度单位其余可类推)
xmajorLocator = MultipleLocator(4);

# 设置x轴标签文本的格式
xmajorFormatter = FormatStrFormatter('%d') 

# 将x轴次刻度标签设置为5的倍数
xminorLocator = MultipleLocator(1) 

# 设定y 轴的主刻度间隔及相应的刻度间隔显示格式

# 将y轴主刻度标签设置为1.0的倍数
ymajorLocator = MultipleLocator(0.005)

#  设置y轴标签文本的格式
ymajorFormatter = FormatStrFormatter('%1.3f')

# 将此y轴次刻度标签设置为0.2的倍数
yminorLocator = MultipleLocator(0.001)

x = [1, 2, 3, 4]
# y_www = [0.1492, 0.1425, 0.1332, 0.1251]
# y_kdd = [0.1504, 0.1441, 0.1283, 0.1247]
y_www = [0.3705, 0.3313, 0.3018, 0.2984]
y_kdd = [0.3221, 0.3155, 0.2792, 0.2751]

www_plot = plt.subplot(111)
plt.title('MRR')
plt.plot(x, y_www, '-bo', label='WWW')
plt.plot(x, y_kdd, '-rD' ,label='KDD')
# plt.plot(x, y_f1_w, '-gs', label='F1-score')
ax = plt.gca()
ax.axis([0, 5, 0.27, 0.38]) 
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

plt.show()