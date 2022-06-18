# Learners：Sunny
# university：SWUFE
# school：RIEM
# 开发时间：2022-06-07 7:49
import jieba


# 待分词的文本路径
sourceTxt = 'source01.txt'
# 分好词后的文本路径
targetTxt = 'target.txt'

# 对文本进行操作
with open(sourceTxt, 'r', encoding='utf-8') as sourceFile, open(targetTxt, 'a+', encoding='utf-8') as targetFile:
        for line in sourceFile:
            seg = jieba.cut(line.strip(), cut_all=False)
            # 分好词之后之间用/隔断
            output = '/'.join(seg)
            targetFile.write(output)
            targetFile.write('\n')
        print('写入成功！')
        sourceFile.close()
        targetFile.close()