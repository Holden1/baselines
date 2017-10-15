from collections import deque

import numpy as np
from baselines.common.atari_wrappers_deprecated import LazyFrames
from grabber import Grabber
import cv2
import time
import os
import sys
from directkeys import W,A,S,D,P,U,E,Q,T,L,F1,F2,F3,NUM1,NUM2,NUM4,SPACE,PressKey,ReleaseKey,PressAndRelease
from numpy import genfromtxt
from windowMgr import WindowMgr
import os
import subprocess

BOSSAREA=400100
BONFIREAREA=400101
FRAME_DIFF=0.02

iterations=0
start_time=-1
bossHpLastFrame=sys.maxsize
charHpLastFrame=-sys.maxsize

class dsgym:
    def __init__(self):
        self.frames = deque([], maxlen=4)
        self.fill_frame_buffer=True

    def pause_wrapper(self):
        PressAndRelease(P)

    def unpause_wrapper(self):
        PressAndRelease(U)

    def notresponding(self,name):
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

    def setDsInFocus(self):
        self.releaseAll()
        w=WindowMgr()
        w.find_window_wildcard(".*ARK SOULS.*")
        try:
            w.set_foreground()
        except:
            print("Had issues setting to foreground")

    def CheckAndHandleNotResponding(self):
        #Cheat engine might not be responding if it fails to attach debugger
        if(self.notresponding("cheatengine-x86_64.exe")):
            self.releaseAll()
            os.system("taskkill /f /im  cheatengine-x86_64.exe /T")
            os.system('".\\DarkSoulsIII.CT"')
            time.sleep(5)
            PressAndRelease(T)
            self.setDsInFocus()
        if(self.notresponding("DarkSoulsIII.exe")):
            self.releaseAll()
            print("Game not responding, waiting 5 seconds until restart")
            time.sleep(5)
            PressAndRelease(U)
            if  self.notresponding("DarkSoulsIII.exe"):
                self.kill_processes()
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
                    [ludexHp,charHp,stamina,area,targetLock]=self.readState()

                    if(area==BONFIREAREA):
                        iter=100 #we are in game
                
                time.sleep(5)
                print("Assuming in game now")
                ReleaseKey(E)
                #Set terminal true, this makes us tele to boss
                return True
        else:
            return False

    def teleToBoss(self):
        self.setDsInFocus()
        time.sleep(2)
        for i in range(10):
            PressAndRelease(F1)
            time.sleep(1)
            PressAndRelease(E)
            PressAndRelease(E)#Twice, bloodstain can be at entrance
            time.sleep(2)
            #Check whether we have entered boss area
            [ludexHp,charHp,stamina,area,targetLock]=self.readState()
            if(area==BOSSAREA):
                PressAndRelease(F2)
                PressAndRelease(Q)
                break
        else:   #For loop else, not if else
                #didn't get to boss area in 10 tries, commit sudoku
            PressAndRelease(F3)
            print("Suicide, something wrong")
        

    def readState(self):
        hasRead=False
        while (hasRead==False):
            try:
                [ludexHp,charHp,stamina,area,targetLock] = genfromtxt('gameInfo.txt', delimiter=',')
                hasRead = True
            except:
                print ("Oops couldn't read")
        
        #print("LudexHp:",ludexHp,"charhp",charHp,"Stamina",stamina)
        return ludexHp,charHp,stamina,area,targetLock
    def reset(self):
        self.setDsInFocus()
        self.releaseAll()
        self.teleToBoss()
        self.setDsInFocus()

    def can_reset(self):
        self.releaseAll()
        [ludexHp,charHp,stamina,area,targetLock]=self.readState()
        self.CheckAndHandleNotResponding()
        return charHp !=0

    def kill_processes(self):
        os.system("taskkill /f /im  DarkSoulsIII.exe /T")
        # also kill cheat engine
        os.system("taskkill /f /im  cheatengine-x86_64.exe /T")


    def frame_step(self,input_actions):
        terminal=False
        reward=0

        #stupid python

        global bossHpLastFrame
        global charHpLastFrame
        global iterations

        iterations+=1
        #Todo, fix this by restarting ds? 
        if iterations%1000==1:
            terminal=self.CheckAndHandleNotResponding()

        #Retrieve state BossHp, CharHp and stamina
        [ludexHp,charHp,stamina,area,targetLock]=self.readState()

        #Check if we died
        if(charHp==0 or area==BONFIREAREA):
            #Unpause game and wait for hp>0
            self.releaseAll()
            PressAndRelease(U)
            terminal=True
            reward=-1
        #Check if we killed the boss
        elif ludexHp==0:
            self.releaseAll()
            PressAndRelease(U)
            PressAndRelease(F3)
            terminal=True
            reward=1
        #Check if lost target on boss
        elif targetLock==0:
            PressAndRelease(Q)

        ludexMaxHp=1037
        charMaxHp=454

        ludexNorm=ludexHp/ludexMaxHp
        charNorm=charHp/charMaxHp

        

        self.releaseAll()
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
        if input_actions==7:
            PressKey(NUM4)
        #Input action 8 is doing nothing


        if not terminal:
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
            #PressKey(P)
            #ReleaseKey(P)

        self.add_frames(1)

        if terminal:
            self.fill_frame_buffer=True #Fill buffer next time, if we died




        bossHpLastFrame=ludexHp
        charHpLastFrame=charHp

        

        return LazyFrames(list(self.frames)), reward, terminal

    def releaseAll(self):
        ReleaseKey(P)
        ReleaseKey(W)
        ReleaseKey(A)
        ReleaseKey(S)
        ReleaseKey(D)
        ReleaseKey(E)
        ReleaseKey(SPACE)
        ReleaseKey(NUM1)
        ReleaseKey(NUM2)
        ReleaseKey(NUM4)

    def add_frames(self,num_frames):
        global start_time
        for i in range(num_frames):
            # Sleep to ensure consistency in frames
            if start_time != -1:
                elapsed = time.time() - start_time
                # print(elapsed)
                timeToSleep = FRAME_DIFF - elapsed
                if timeToSleep > 0:
                    time.sleep(timeToSleep)
                    # print("New elapsed ",time.time()-start_time)

            grabber = Grabber(bbox=(8, 40, 808, 450))
            screen = grabber.grab()
            start_time = time.time()
            grayscale_small = cv2.cvtColor(cv2.resize(screen, (84, 84), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
            # millis = int(round(time.time() * 1000))
            # cv2.imwrite('images/'+str(millis)+'grey.png', grayscale_small)
            # if nothing in frame buffer, fill it with this frame

            grayscale_small=np.reshape(grayscale_small, [84, 84, 1])
            if self.fill_frame_buffer:
                for _ in range(4):
                    self.frames.append(grayscale_small)
                self.fill_frame_buffer = False
            else:
                self.frames.append(grayscale_small)