# Learners：Sunny
# university：SWUFE
# school：RIEM
# 开发时间：2022-06-07 10:25
import jieba
import re


def txt():  # 输出词频前N的词语
    txt = open("target.txt", "r", encoding='utf-8').read()  # 打开txt文件,要和python在同一文件夹
    # print(txt)
    txt00 = open("shuchu.txt", "a+", encoding='utf-8')
    words = jieba.lcut(txt)  # 精确模式，返回一个列表
    # print(words)
    counts = {}  # 创建字典
    lt=['三炮','##','......','24','10','30','2020','14','31','11','13','20','15','28','17','16','29','微博']
    stopkeyword = [line.strip() for line in open('stopwords.txt', encoding='utf-8').readlines()]  # 加载停用词
    for word in words:
        if len(word) == 1:
            continue
        elif word  in stopkeyword :
            rword = " "
        else:
            rword = word
        counts[rword] = counts.get(rword, 0) + 1  # 字典的运用，统计词频
    items = list(counts.items())  # 返回所有键值对
    print(items)
    items.sort(key=lambda x: x[1], reverse=True)  # 降序排序
    N = eval(input("请输入N：代表输出的数字个数："))
    wordlist = list()
    r1 = re.compile(r'\w')  # 字母，数字，下划线，汉字
    r2 = re.compile(r'[^\d]')  # 排除数字
    r3 = re.compile(r'[\u4e00-\u9fa5]')  # 中文
    r4 = re.compile(r'[^_]')  # 排除_
    # stopkeyword = [line.strip() for line in open('stopwords.txt', encoding='utf-8').readlines()]  # 加载停用词
    for i in range(N):
        word, count = items[i]
        txt00.write("{0:<10}{1:<5}".format(word, count))  # 输出前N个词频的词语
        txt00.write('\n')
        if  r1.match(word) and r2.match(word) and r3.match(word) and r4.match(word) :
            continue

    txt00.close()


# 调用词频统计函数
txt()