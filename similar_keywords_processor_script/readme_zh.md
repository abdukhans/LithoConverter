# READMEN

我要写一个python 脚本, 完成对一个csv文件的阅读,删除行, 合并行, 写新csv文件的功能!
细化需求如下:

1. 创建 文本数据库(similarKeywords)
数据库只有一张表
表名: similarKeywords
字段
Attribute txt
Frequency number
Similar_Words txt
rm, boolen, 默认false
combine, number, 默认 0


2. 读取指定文件名为 original.csv文件,存入文本数据库中
文件内容:
Attribute,Frequency,Similar_Words
igneous,188361,"[('igneous', 188361)]"
sedimentary,93604,"[('sedimentary', 93604)]"
metamorphic,69654,"[('metamorphic', 69654)]"

3. 读取指定文件名task.csv 文件, 
文件内容:
task, word
rm, "rock"
combine, "rhyolite, schist"

根据文件中的taks, 对similarKeywords表中记录做标记
task1: rm rock
在表中 Attribute = “rock” 查找, 在找出记录的 rm = true

task2:
combine, "rhyolite, schist"
计数 combineNum, 在第一条combine task, 记为1,每次加一 
在表中, 一次找出word 字短中的word(Attribute = “rock”),
另每一个word的记录中的 combine = combineNum

3. 创建新文件 new.csv 写入第一行内容:
Attribute,Frequency,Similar_Words

4. 将表similarKeywords中的容写入new.csv
1) 过滤出 combine==0 && tm==false 的记录
原样写入new.csv

2) 过滤出 combine 的记录
遍历规则 for(i=1,i<=combineNum,i++ )

讲combineNum 相同的记录过滤出来, 做合并操作
Attribute = 第一条记录的Attribute
Frequency = 所有记录的Frequency 之和
Similar_Words = 将每一条记录的Similar_Words合并 
例如"[('sedimentary', 93604)]","[('metamorphic', 69654)]", 合并后"[('sedimentary', 93604),('metamorphic', 69654)]"
最后, 讲合并后的记录,写入new.csv
