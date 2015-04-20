Voice Music Separation
--------

该工程用于参加第六届浙大华为杯题目：
[流行歌曲歌声提取](http://paas-developer.huawei.com/competition#!/competition/subjects/551234f01bd7a2a52753d995)

主要功能为歌声音乐分离'voiceMusicSeparation()'以及播放音乐'playAudio()'，还有内部函数矩阵恢复的IALM算法等。

--------

该工程基于Python开发，需要安装Python及工程相关包：NumPy、SciPy、Pymedia、stft

使用步骤

1. 下载并解压工程，解压后路径如'E:Python\voiceMusicSeparation'
2. 打开Python，把工程文件夹加入到搜索路径
```
import sys
sys.path.append('E:\Python\voiceMusicSeparation')
```
如果要永久添加该路径，可以在Python的'D:\Python27\Lib\site-packages'文件夹下新建'mypkpath.pth'，里面写上要添加的路径
```
# .pth file for my project(这行是注释)
E:\Python\voiceMusicSeparation
```
3. 导入该工程module
```
import voiceMusicSeparation as vms
```
4. 把你的wav格式歌曲放到Audio文件夹，注意歌曲名称改成英文。进行歌声和音乐的分离
```
vms.voiceMusicSeparation('Audio/歌曲名称.wav')
```
5. 打开工程主目录下生成的歌声'outputE.wav'和音乐'outputA.wav'查看分离效果。也可以在Python中播放：
```
vms.playAudio('outputE.wav')
```

- - -

作者：Luwak队@ZJU

说明：仅用于学习交流，禁止用于商业目的
