from collections import deque

import numpy as np
import win32ui
from baselines.common.atari_wrappers_deprecated import LazyFrames
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


BOSSAREA=400100
BONFIREAREA=400101
FRAME_DIFF=0.02

iterations=0
start_time=-1
bossHpLastFrame=sys.maxsize
charHpLastFrame=-sys.maxsize
not_responding_lock=threading.Lock()

class dsgym:
    def __init__(self):
        self.frames = deque([], maxlen=4)
        self.prev_features = deque([], maxlen=4)

        self.fill_frame_buffer=True
        self.spawnCheckRespondingThread()

    def pause_wrapper(self):
        PressAndRelease(P)
    def speed_up_wrapper(self):
        PressAndRelease(I)
    def normal_speed_wrapper(self):
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
                        os.system('"C:\Program Files (x86)\Steam\steamapps\common\DARK SOULS III\Game\DarkSoulsIII.exe"')
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
                            [_,_,_,area,*_]=self.readState()

                            if(area==BONFIREAREA):
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
            [_,_,_,area,*_]=self.readState()
            if(area==BOSSAREA):
                PressAndRelease(F2)
                PressAndRelease(Q)
                break
        else:   #For loop else, not if else
                #didn't get to boss area in 10 tries, commit sudoku and kill both processes
            PressAndRelease(F3)
            print("Couldn't get to boss in 20 tries, something wrong, killing processes as well")
            self.kill_processes()
        

    def readState(self):
        hasRead=False
        while (hasRead==False):
            try:
                data = genfromtxt('gameInfo.txt', delimiter=',')
                assert len(data) >0
                hasRead = True
            except:
                print ("Oops couldn't read")
        
        #print("LudexHp:",ludexHp,"charhp",charHp,"Stamina",stamina)
        return data
    def reset(self):
        self.setDsInFocus()
        self.releaseAll()
        self.teleToBoss()
        self.setDsInFocus()

    def can_reset(self):
        self.releaseAll()
        [_,charHp,*_]=self.readState()
        #self.CheckAndHandleNotResponding()
        return charHp !=0

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

        #Check if able to take not responding lock
        self.check_responding_lock()

        #Retrieve state BossHp, CharHp and stamina
        [ludexHp,charHp,stamina,area,targetLock,*feats]=self.readState()

        #Check if we died
        if(charHp==0 or area==BONFIREAREA):
            #Unpause game and wait for hp>0
            self.releaseAll()
            PressAndRelease(U)
            terminal=True
            reward=-1
        #Check if we killed the boss
        if ludexHp==0:
            self.releaseAll()
            PressAndRelease(U)
            PressAndRelease(F3)
            terminal=True
            reward=1
        #Check if lost target on boss
        elif targetLock==0:
            PressAndRelease(Q)

        #ludexMaxHp=1037
        #harMaxHp=454

        self.releaseAll()
        #Input action
        #Handle movement
        movement=input_actions//5

        if movement == 0:
            PressKey(W)
        if movement == 1:
            PressKey(A)
        if movement == 2:
            PressKey(S)
        if movement == 3:
            PressKey(D)
        #Handle actions
        action = input_actions % 5
        if action == 0:
            PressKey(SPACE)
        if action == 1:
            PressKey(NUM1)
        if action == 2:
            PressKey(NUM2)
        if action==3:
            PressKey(NUM4)
        #Other actions do nothing

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
        self.add_features(input_actions,[ludexHp,charHp,stamina,*feats])
        if terminal:
            self.fill_frame_buffer=True #Fill buffer next time, if we died
            PressAndRelease(I) #speed up when dead
        bossHpLastFrame=ludexHp
        charHpLastFrame=charHp
        return LazyFrames(list(self.frames)), np.hstack(self.prev_features), reward, terminal

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

    def add_features(self, action_to_add,features_to_add):
        action_one_hot=np.zeros(25)
        action_one_hot[action_to_add]=1

        total_feature_vector=np.nan_to_num(np.hstack((action_one_hot,features_to_add)))

        if self.fill_frame_buffer:
            for _ in range(4):
                self.prev_features.append(total_feature_vector)
            self.fill_frame_buffer = False
        else:
            self.prev_features.append(total_feature_vector)


