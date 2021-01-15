from collections import deque

import numpy as np
import win32ui
from baselines.common.atari_wrappers import LazyFrames
from grabber import Grabber
import cv2
import time
import os
import sys
from directkeys import W,A,S,D,P,U,E,Q,T,L,I,F1,F2,F3,NUM1,NUM2,NUM4,SPACE,PressKey,ReleaseKey,PressAndRelease,PressAndFastRelease
from numpy import genfromtxt
from windowMgr import WindowMgr
import os
import subprocess
import threading
from gym import spaces
import math


BOSSAREA="400100"
BONFIREAREA="400101"
FRAME_DIFF=0.02

iterations=0
start_time=-1
bossHpLastFrame=sys.maxsize
charHpLastFrame=-sys.maxsize
not_responding_lock=threading.Lock()

areaKey="locationArea"
charHpKey="heroHp"
bossHpKey="targetedEntityHp"

num_state_scalars=26
num_history_states=100


def parse_val(value):
    try:
        val=float(value)
        return val
    except ValueError:
        if value=="??":
            return 0
        return value

class dsgym:
    
    observation_space=spaces.Box(-100,1000,num_history_states*num_state_scalars)
    action_space=spaces.Discrete(9)
    def __init__(self):
        self.frames = deque([], maxlen=4)
        self.prev_actions = deque([], maxlen=4)

        
        self.prev_state = deque([], maxlen=num_history_states)

        self.fill_frame_buffer=True
        self.spawnCheckRespondingThread()
        self.logfile = open("gameInfo.txt", "r", encoding="utf-8")

        self.paused=False


    def unpause_wrapper(self):
        if(self.paused):
            PressAndFastRelease(U)
            self.paused=False

    def pause_wrapper(self):
        PressAndRelease(P)
        self.paused=True
    def speed_up_wrapper(self):
        PressAndRelease(I)
    def normal_speed_wrapper(self):
        PressAndFastRelease(U)

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
    def spawnCheckRespondingThread(self):
        thread = threading.Thread(target=self.CheckAndHandleNotResponding, args=())
        thread.daemon = True  # Daemonize thread
        thread.start()  # Start the execution

    def window_exists(self,window_name):
        try:
            win32ui.FindWindow(None, window_name)
            return True
        except win32ui.error:
            return False
    def CheckAndHandleNotResponding(self):
        while True:
            #Cheat engine might not be responding if it fails to attach debugger
            if(self.notresponding("cheatengine-x86_64.exe") or self.window_exists("Lua Engine")):
                with not_responding_lock:
                    self.releaseAll()
                    os.system("taskkill /f /im  cheatengine-x86_64.exe /T")
                    os.system('".\\DarkSoulsIII.CT"')
                    time.sleep(5)
                    PressAndRelease(T)
                    PressAndRelease(T)
                    self.setDsInFocus()
            if(self.notresponding("DarkSoulsIII.exe") or self.window_exists("Error")):
                with not_responding_lock:
                    self.releaseAll()
                    print("Game not responding, waiting 5 seconds until restart")
                    PressAndRelease(U)
                    time.sleep(5)
                    if (self.notresponding("DarkSoulsIII.exe")or self.window_exists("Error")):
                        self.kill_processes()
                        os.system('".\\DarkSoulsIII.CT"')
                        time.sleep(5)
                        os.system('"F:\SteamLibrary\steamapps\common\DARK SOULS III\Game\DarkSoulsIII.exe"')
                        w=WindowMgr()
                        time.sleep(40)
                        PressAndRelease(T)
                        PressAndRelease(I)
                        w.find_window_wildcard(".*ARK SOULS.*")
                        iter=0
                        while iter<1000:
                            try:
                                w.set_foreground()
                            except:
                                print("Had issues setting to foreground")

                            print("Spamming E to get into game",iter)
                            PressAndFastRelease(E)
                            iter+=1
                            stateDict=self.readState()

                            if(stateDict[areaKey]==BONFIREAREA):
                                break #we are in game

                        time.sleep(5)
                        print("Assuming in game now")
                        PressAndRelease(T)
                        ReleaseKey(E)
            time.sleep(5)


    def teleToBoss(self):
        self.setDsInFocus()
        time.sleep(5)
        for i in range(20):
            self.check_responding_lock()
            PressAndRelease(F1)
            PressAndRelease(U)#Normal speed
            PressAndRelease(E)
            PressAndRelease(E)#Twice, bloodstain can be at entrance
            time.sleep(2)
            #Check whether we have entered boss area
            stateDict=self.readState()
            if(stateDict[areaKey]==BOSSAREA):
                PressAndRelease(F2)
                PressAndRelease(Q)
                break
        else:   #For loop else, not if else
                #didn't get to boss area in 10 tries, commit sudoku and kill both processes
            PressAndRelease(F3)
            print("Couldn't get to boss in 20 tries, something wrong, killing processes as well")
            self.kill_processes()

    def kill_or_wait(self,start_read):
        elapsed = int(time.time() - start_read)
        max_wait_time = 30
        print("waiting for loading screen", elapsed, " of max", max_wait_time)
        if elapsed >= max_wait_time:
            self.kill_processes()
            # wait for restart thread to pick it up, then wait for lock
            time.sleep(10)
            self.check_responding_lock()
        else:
            time.sleep(1)



    def readState(self):
        hasRead=False
        start_read=time.time()

        while (hasRead==False):
            self.logfile.seek(0)
            loglines = self.logfile.readline()
            if not loglines or len(loglines.split(";;"))<4:
                continue
            stateDict= {}
            for line in loglines.split(";;"):
                try:
                    (key,val) = line.split("::")
                    stateDict[key]=val
                except:
                    continue

            hasRead = True
        return stateDict
    def reset(self):
        self.setDsInFocus()
        self.releaseAll()
        self.teleToBoss()
        self.setDsInFocus()

    def can_reset(self):
        self.releaseAll()
        stateDict=self.readState()
        #self.CheckAndHandleNotResponding()
        return stateDict[charHpKey] !=0

    def kill_processes(self):
        os.system("taskkill /f /im  DarkSoulsIII.exe /T")
        # also kill cheat engine
        os.system("taskkill /f /im  cheatengine-x86_64.exe /T")
    def check_responding_lock(self):
        not_responding_lock.acquire()
        not_responding_lock.release()

    def frame_step(self,input_actions):
        terminal=False
        reward=0

        #stupid python

        global bossHpLastFrame
        global charHpLastFrame
        global iterations

        iterations+=1
        self.unpause_wrapper()
        #Check if able to take not responding lock
        
        self.check_responding_lock()

        stateDict = self.readState()
        #Check if we died
        if(stateDict[charHpKey]=="0" or stateDict[areaKey]==BONFIREAREA or stateDict[areaKey]=="??"):
            #Unpause game and wait for hp>0
            self.releaseAll()
            PressAndRelease(U)
            terminal=True
            reward=-1
        #Check if we killed the boss or missing boss into
        elif stateDict[bossHpKey]==0 or stateDict[bossHpKey]=="??":
            self.releaseAll()
            print("killed boss")
            PressAndRelease(U)
            PressAndRelease(F3)
            terminal=True
            reward=1
        #Check if lost target on boss
        elif stateDict["targetLock"]==0:
            PressAndRelease(Q)

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
            if bossHpLastFrame>int(stateDict[bossHpKey]):
                reward+=1
            #If our hp is lower than last frame
            elif int(stateDict[charHpKey])<int(charHpLastFrame):
                reward-=1

            bossHpLastFrame=int(stateDict[bossHpKey])
            charHpLastFrame=int(stateDict[charHpKey])


        self.add_frames(1)
        self.add_state(input_actions,stateDict)
        if terminal:
            self.fill_frame_buffer=True #Fill buffer next time, if we died
            PressAndRelease(I) #speed up when dead
        
        return np.hstack(self.prev_state), reward, terminal

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
                    #print("New elapsed ",time.time()-start_time)

            grabber = Grabber(bbox=(8, 40, 808, 450))
            screen = grabber.grab()
            start_time = time.time()
            grayscale_small = cv2.cvtColor(cv2.resize(screen, (119, 70), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
            #millis = int(round(time.time() * 1000))
            #cv2.imwrite('images/'+str(millis)+'grey.png', grayscale_small)
            # if nothing in frame buffer, fill it with this frame

            grayscale_small=np.reshape(grayscale_small, [119, 70, 1])
            if self.fill_frame_buffer:
                for _ in range(4):
                    self.frames.append(grayscale_small)
                #self.fill_frame_buffer = False done with actions instead
            else:
                self.frames.append(grayscale_small)

    def add_actions(self,action_to_add):
        action_one_hot=np.zeros(9)
        action_one_hot[action_to_add]=1

        if self.fill_frame_buffer:
            for _ in range(4):
                self.prev_actions.append(action_one_hot)
            self.fill_frame_buffer = False
        else:
            self.prev_actions.append(action_one_hot)

    def parseStateDictValue(self,stateDict,key):
        if (stateDict[key]=="??"):
            return 0
        else:
            return float(stateDict[key])
    def calc_dist(self,targetx,targety,herox,heroy):
        return math.sqrt((targetx-herox)**2+(targety-heroy)**2)

    def add_state(self,action_to_add,stateDict):
        targetX=self.parseStateDictValue(stateDict,"targetedEntityX")
        targetY=self.parseStateDictValue(stateDict,"targetedEntityY")
        heroX=self.parseStateDictValue(stateDict,"heroX")
        heroY=self.parseStateDictValue(stateDict,"heroY")

        stateToAdd=np.zeros(num_state_scalars)
        stateToAdd[action_to_add]=1
        if(stateDict["targetAnimationName"])=="DamageParryEnemy1":
            stateToAdd[9]=1
        stateToAdd[10]=self.parseStateDictValue(stateDict,"targetedEntityHp")
        stateToAdd[11]=targetX
        stateToAdd[12]=targetY
        stateToAdd[13]=self.parseStateDictValue(stateDict,"targetedEntityZ")
        stateToAdd[14]=self.parseStateDictValue(stateDict,"targetedEntityAngle")
        stateToAdd[15]=self.parseStateDictValue(stateDict,"targetAttack1")
        if(float(self.parseStateDictValue(stateDict,"targetAttack2"))>3000):
            stateToAdd[16]=float(self.parseStateDictValue(stateDict,"targetAttack2"))-3000
        else:
            stateToAdd[16]=float(self.parseStateDictValue(stateDict,"targetAttack2"))
        stateToAdd[17]=self.parseStateDictValue(stateDict,"targetMovement1")
        stateToAdd[18]=self.parseStateDictValue(stateDict,"targetMovement2")
        stateToAdd[19]=self.parseStateDictValue(stateDict,"targetComboAttack")
        stateToAdd[20]=self.parseStateDictValue(stateDict,"heroHp")
        stateToAdd[21]=heroX
        stateToAdd[22]=heroY

        dist=self.calc_dist(targetX,targetY,heroX,heroY)
        print("dist",dist)

        stateToAdd[23]=dist
        stateToAdd[24]=self.parseStateDictValue(stateDict,"heroAngle")
        stateToAdd[25]=self.parseStateDictValue(stateDict,"heroSp")

        print("State",stateToAdd)
        if self.fill_frame_buffer:
            for _ in range(num_history_states):
                self.prev_state.append(stateToAdd)
            self.fill_frame_buffer = False
        else:
            self.prev_state.append(stateToAdd)


