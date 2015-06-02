# -*- coding: utf-8 -*-
"""
voice music separation
--------------------------------------
reference
Po-Sen Huang / Singing-voice separation from monaural recordings using robust principal component analysis
Zhouchen Lin / The augmented lagrange multiplier method for exact recovery of corrupted low-rank matrices
---------------------------------------
used package
math/scipy/numpy
stft0.4.7 URL:https://pypi.python.org/pypi/stft/0.4.7
pymedia1.3.7.3 URL:http://www.lfd.uci.edu/~gohlke/pythonlibs/#pymedia
-------------------------------------
@author: HanSir
website: http://hansir.net
Created on Fri Apr 17 09:00:18 2015
"""
import math
import scipy
import numpy as np
# 播放音频
def playAudio(filename):
    import pymedia.audio.acodec as acodec
    import pymedia.muxer as muxer
    import pymedia.audio.sound as sound
    import time

    name1 = str.split(filename, '.')
    # Open demuxer first
    dm = muxer.Demuxer(name1[-1].lower())
    dec = None
    snd = None
    s = " "
    f = open(filename, 'rb')

    while len(s):
        s = f.read(20000)
        if len(s):
            # 解析出最初的几帧音频数据 
            frames = dm.parse(s)
            for fr in frames:
                if dec == None:
                    # Open decoder
                    dec = acodec.Decoder(dm.streams[0])
                #音频数据在 frame 数组的第二个元素中
                r = dec.decode(fr[1])
                if r and r.data:
                    if snd == None:
                        snd = sound.Output(r.sample_rate, r.channels, sound.AFMT_S16_LE)
                    snd.play(r.data)
    #8.延时，直到播放完毕
    while snd.isPlaying():
        time.sleep(0.5)
        
# 收缩算子
def shrinkage(X,eps):
    S = np.sign(X)*(abs(X)-1.0*eps)*(abs(X)>eps)
    return S
# RPCA的IALM算法    
def ialmRPCA(D, lamb=1, tol1=0.00001, tol2=0.0001, maxIter=1000):
    n1,n2 = np.shape(D)
    lamb = lamb/math.sqrt(max(n1,n2))
    A = np.zeros((n1,n2), dtype=float)
    E = np.zeros((n1,n2), dtype=float)
    S = np.zeros((n1,n2), dtype=float)
    # 受启发于对偶方法，Y=sgn(D)/J(sgn(D))
    normDfro = np.linalg.norm(D,'fro')
    normD2 = np.linalg.norm(D,2)
    normDinf = np.linalg.norm(D,np.inf)
    JD = max(normD2,normDinf)
    Y = np.sign(D)/JD
    # 根据Lin的论文
    mu = 1.25/normD2
    rho = 1.6
    for i in range(maxIter):
        U, s, Vh = scipy.linalg.svd(Y/mu+D-E,full_matrices=True)
        for j in range(len(s)):
            S[j][j] = s[j]
        S = shrinkage(S,1/mu)
        A = np.dot(np.dot(U,S),Vh)
        tempE = E;
        E = shrinkage(Y/mu+D-A,lamb/mu)
        # update Y and mu
        Y += mu*(D-A-E)
        mu = rho*mu
        # dispaly
        stop1 = np.linalg.norm(D-A-E,'fro')/normDfro
        stop2 = np.linalg.norm(E-tempE,'fro')/normDfro
        print("iteration:%d, |D-A-E|_F/|D|_F:%.8f" % (i,stop1))
        # stop
        if(stop1<tol1 and stop2<tol2):
            break;
        if(i>=maxIter):
            print("Max number of iter reached.")
            break;
    return A,E,i

# 歌声音乐分离
def voiceMusicSeparation(audio,masktype=1,lamb=1.25,gain=1.25):
    import stft
    # stft       
    specgram = stft.spectrogram(audio)
    # rpca
    D = abs(specgram)
    angle = np.angle(specgram)
    A_mag,E_mag,numiter = ialmRPCA(D,lamb)
    A = A_mag*scipy.exp(angle*1j)
    E = E_mag*scipy.exp(angle*1j)
    # binary mask
    if(masktype):
        m = 1.0*(abs(E_mag)>abs(gain*A_mag))
        Emask = m*specgram
        Amask = specgram-Emask
    else:
        Emask = E
        Amask = A       
    # istft
    outputA = stft.ispectrogram(Amask)
    outputE = stft.ispectrogram(Emask)
    #output
    wavoutA = np.array(outputA[:len(audio)],dtype=np.int16)   
    wavoutE = np.array(outputE[:len(audio)],dtype=np.int16)
    return wavoutA, wavoutE

# 对歌曲进行歌声音乐分离（单双通道都可以处理）
def fuck(filename,masktype=1,lamb=1.25,gain=1.25):
    import scipy.io.wavfile as wav
    # input
    fs, audio = wav.read(filename)
    print("Voice Music Separation Starts...")
    if(audio.shape[1]==1):
        # voiceMusicSeparation
        wavoutA, wavoutE = voiceMusicSeparation(audio,masktype,lamb,gain)
        # output
        wav.write('outputA.wav', fs, wavoutA)
        wav.write('outputE.wav', fs, wavoutE)
    elif(audio.shape[1]==2):
        audio0 = np.array([audio[i][0] for i in range(len(audio))])
        audio1 = np.array([audio[i][1] for i in range(len(audio))])
        # voiceMusicSeparation
        wavoutA0, wavoutE0 = voiceMusicSeparation(audio0,masktype,lamb,gain)
        wavoutA1, wavoutE1 = voiceMusicSeparation(audio1,masktype,lamb,gain)
        # output
        wavoutA = np.array([[wavoutA0[i],wavoutA1[i]] for i in range(len(audio))])
        wav.write('outputA.wav', fs, wavoutA)
        wavoutE = np.array([[wavoutE0[i],wavoutE1[i]] for i in range(len(audio))])
        wav.write('outputE.wav', fs, wavoutE)
    else:
        print("Sorry, your song is too complex to deal with.")
    print("Voice Music Separation Completes.")
    print("voice: outputE.wav")
    print("music: outputA.wav")
##########################################################################33
#filename = 'Audio\Better Man_clip.wav'
#masktype = 1 # 1: binary mask, 0: no mask
#lamb = 1.25 # lambda in ialm_rpca
#gain = 1.25
#fuck(filename)
