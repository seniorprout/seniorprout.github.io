# Learners：Sunny
# university：SWUFE
# school：RIEM
# 开发时间：2022-06-07 7:47
# -*- coding: UTF-8 -*-

readPath='source.txt'
writePath='source01.txt'
lines_seen=set()
outfiile=open(writePath,'a+',encoding='utf-8')
f=open(readPath,'r',encoding='utf-8')
for line in f:
    if line not in lines_seen:
        outfiile.write(line)
        lines_seen.add(line)