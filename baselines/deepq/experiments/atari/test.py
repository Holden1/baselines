import gym
import psutil
import time
start_time = time.time()
import subprocess

import os

def getTasks(name):
    r = os.popen('tasklist /v').read().strip().split('\n')
    #print ('# of tasks is %s' % (len(r)))
    for i in range(len(r)):
        s = r[i]
        if name in r[i]:
            #print ('%s in r[i]' %(name))
            return r[i]
    
    return []

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

imgName = 'DarkSoulsIII.exe'
notResponding = 'Not Responding'
for i in range(100):
    print(notresponding(imgName))

print("--- %s seconds ---" % (time.time() - start_time))