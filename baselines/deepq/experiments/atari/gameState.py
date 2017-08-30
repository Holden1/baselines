import numpy as np
from grabber import Grabber
import cv2
import time
import os
import sys
from directkeys import W,A,S,D,P,U,E,Q,T,L,F1,F2,NUM1,NUM2,SPACE,PressKey,ReleaseKey,PressAndRelease
from numpy import genfromtxt
from windowMgr import WindowMgr
import os
import subprocess

iterations=0
start_time=-1
bossHpLastFrame=sys.maxsize
charHpLastFrame=-sys.maxsize

def notresponding(name):
    #os.system('tasklist /FI "IMAGENAME eq %s" /FI "STATUS eq not responding" > tmp.txt' % name)
    #x = subprocess.check_output()
    a = subprocess.Popen('tasklist /FI "IMAGENAME eq %s" /FI "STATUS eq running"' % name,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    a=a.communicate()[0].decode("utf-8")
    b = subprocess.Popen('tasklist /FI "IMAGENAME eq WerFault.exe" /FI "STATUS eq running"',stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    b=b.communicate()[0].decode("utf-8")
    c = subprocess.Popen('tasklist /FI "IMAGENAME eq %s" /FI "STATUS ne running"' % name,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    c=c.communicate()[0].decode("utf-8")
    #tmp.close()
    if c.split("\n")[-2].startswith(name) or "INFO:" not in b:
        return True
    elif a.split("\n")[-2].startswith(name):
        return False
    else:
        return True

def setDsInFocus():
    releaseAll()
    w=WindowMgr()
    w.find_window_wildcard(".*ARK SOULS.*")
    try:
        w.set_foreground()
    except:
        print("Had issues setting to foreground")

def CheckAndHandleNotResponding():
    #Cheat engine might not be responding if it fails to attach debugger
    if(notresponding("cheatengine-x86_64.exe")):
        releaseAll()
        os.system("taskkill /f /im  cheatengine-x86_64.exe /T")
        os.system('".\\DarkSoulsIII.CT"')
        time.sleep(5)
        PressAndRelease(T)
        setDsInFocus()
    if(notresponding("DarkSoulsIII.exe")):
        releaseAll()
        print("Game not responding, waiting 5 seconds until restart")
        time.sleep(5)
        if  notresponding("DarkSoulsIII.exe"):
            os.system("taskkill /f /im  DarkSoulsIII.exe /T")
            #also kill cheat engine
            os.system("taskkill /f /im  cheatengine-x86_64.exe /T")
            os.system('".\\DarkSoulsIII.CT"')
            time.sleep(5)
            os.system('"C:\Program Files (x86)\Steam\steamapps\common\DARK SOULS III\Game\DarkSoulsIII.exe"')
            w=WindowMgr()         
            time.sleep(40)
            PressAndRelease(T)
            w.find_window_wildcard(".*ARK SOULS.*")
            iter=0
            while iter<100:
                try:
                    w.set_foreground()
                except:
                    print("Had issues setting to foreground")

                print("Spamming E to get into game",iter)
                PressAndRelease(E)
                iter+=1
                [ludexHp,charHp,stamina,area]=readState()

                if(area==400101):
                    iter=100 #we are in game
            
            time.sleep(5)
            print("Assuming in game now")
            ReleaseKey(E)
            #Set terminal true, this makes us tele to boss
            return True
    else:
        return False

def teleToBoss():
    setDsInFocus()
    time.sleep(2)
    PressAndRelease(F1)
    time.sleep(1)
    PressAndRelease(E)
    PressAndRelease(E)
    time.sleep(3)
    PressAndRelease(F2)
    PressAndRelease(Q)

def readState():
    hasRead=False
    while (hasRead==False):
        try:
            [ludexHp,charHp,stamina,area] = genfromtxt('gameInfo.txt', delimiter=',')
            hasRead = True
        except:
            print ("Oops couldn't read")
    
    #print("LudexHp:",ludexHp,"charhp",charHp,"Stamina",stamina)
    return ludexHp,charHp,stamina,area

def frame_step(input_actions):
    terminal=False
    reward=0

    #stupid python
    global start_time
    global bossHpLastFrame
    global charHpLastFrame
    global iterations

    iterations+=1
    #Todo, fix this by restarting ds? 
    if iterations%1000==1:
        terminal=CheckAndHandleNotResponding()
        

    # if start_time!=-1:
    #     elapsed=time.time() - start_time
    #     timeToSleep=0.05-elapsed
    #     if timeToSleep>0:
    #         time.sleep(timeToSleep)
            #print("sleeping")

    
    

    #Retrieve BossHp, CharHp and stamina
    [ludexHp,charHp,stamina,area]=readState()

    #Check if we died
    if(charHp==0 or area==400101):
        #Unpause game and wait for hp>0
        releaseAll()
        PressAndRelease(U)
        terminal=True
        reward=-1
        loaditer=0
        while charHp==0 and loaditer<30:
            loaditer+=1
            #print("Waiting for loading screen ...")
            #Make sure game is still responding
            CheckAndHandleNotResponding()
            time.sleep(0.5)
            [ludexHp,charHp,stamina,area]=readState()
        
        time.sleep(5)
        setDsInFocus()
    

    ludexMaxHp=1037
    charMaxHp=454

    ludexNorm=ludexHp/ludexMaxHp
    charNorm=charHp/charMaxHp


    #If we died and are alive again, teleport to gate go in and teleport to boss
    if(terminal):
        releaseAll()
        teleToBoss()
    else:
        #reward+=charNorm
        #reward-=ludexNorm
        #If we dealt damage to boss since last frame_step
        if bossHpLastFrame>ludexHp:
            #Give reward according to %damage dealt
            #reward+=(bossHpLastFrame-ludexHp)/ludexMaxHp
            reward+=1
        #If our hp is lower than last frame
        elif charHp<charHpLastFrame:
            #Lose reward accoring to %health lost
            #reward-=(charHpLastFrame-charHp)/charMaxHp
            reward-=1
        #high value of stamina means we've run out
        #Penalize running out of stamina
        #Outcommented for now
        #if(stamina==0 or stamina>10000):
        #   reward-=1

    releaseAll()
    #Input action
    if input_actions == 0:
        PressKey(W)
    if input_actions == 1:
        PressKey(A)
    if input_actions == 2:
        PressKey(S)
    if input_actions == 3:
        PressKey(D)
    if input_actions == 4:
        PressKey(SPACE)
    if input_actions == 5:
        PressKey(NUM1)
    if input_actions == 6:
        PressKey(NUM2)
    # if input_actions[7] == 1: #Jump right
    #     PressKey(D)
    #     PressKey(SPACE)
    # if input_actions[8] == 1:#Jump left
    #     PressKey(A)
    #     PressKey(SPACE)  

    start_time = time.time()
    PressKey(P)
    ReleaseKey(P)
    

    grabber = Grabber(bbox=(8, 40, 808, 450))
    screen = grabber.grab()

    bossHpLastFrame=ludexHp
    charHpLastFrame=charHp

    return screen, reward, terminal

def releaseAll():
    ReleaseKey(P)
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    ReleaseKey(E)
    ReleaseKey(SPACE)
    ReleaseKey(NUM1)
    ReleaseKey(NUM2)