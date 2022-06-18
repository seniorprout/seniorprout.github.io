#!/usr/bin/python
# -*- coding: UTF-8 -*-

import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.figure(figsize=(6, 6))
label = ['积极', '消极', '客观']
explode = [0.01, 0.01, 0.01]
values = [2250, 274, 3257]
plt.pie(values, explode=explode, labels=label, autopct='%1.1f%%')
plt.title('情感倾向饼状图')
plt.savefig('./情感倾向饼状图')
plt.show()
