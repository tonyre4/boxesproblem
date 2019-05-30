#Dimensions
#Big Box: w:30 l:50 h:25
#Boxes: 
    #1: 1x2x1
    #2: 2x2x1
    #3: 2x3x1
    #4: 5x4x2
    #5: 5x1x2
import matplotlib.pyplot as plt
import numpy as np

class boxes2D:
    def __init__(self):
        b1  = np.array([1,2])
        b2  = np.array([2,2])
        b3  = np.array([2,3])
        b4  = np.array([5,4])
        b5  = np.array([5,1])
        self.b = np.vstack([b1,b2,b3,b4,b5])

    def getBox(self):
        idx = np.random.randint(0,high=5)
        return idx+2,self.b[idx]

    def printbs(self):
        print(self.b)

    def idxBox(self,idx):
        idx -= 2
        return self.b[idx]


class entity2D:
    def __init__(self):
        #MATRIX
        self.l = 20
        self.w = 30
        self.cntbxs = np.zeros(5,dtype=np.uint8)
        self.unfilled = 0
        self.box = np.zeros((self.l,self.w),dtype=np.uint8)

    #Coding:
        #0: unfilled
        #1: filled
        #2-6: boxtype (filled left-to-right and bottom-to-top)

    def solveFull(self,bxs):
        for l in range(self.l):
            for w in range(self.w):
                if self.box[l,w] == 0: #if unit is not filled
                    idx,rndbox = bxs.getBox() #Get random box
                    if self.itBoxFit((l,w),rndbox): #If box can be set
                        #Fill with the box
                        self.setBox((l,w),idx,rndbox)

        self.getFeatures()
        #self.printbox()
        #print(self.unfilled)
        #print(self.cntbxs)
        #exit()

    def itBoxFit(self,coords,box):
        lo,wo = coords #origin coords
        for w in range(box[0]):
            for l in range(box[1]):
                try:
                    if self.box[lo+l,wo+w] == 0:
                        pass
                    else:
                        return False
                except:
                    return False
        return True

    def printbox(self):
        b = np.flip(self.box,0)
        print (b,"\n")

    def getFeatures(self):
        self.unfilled= np.count_nonzero(self.box==0)
        self.cntbxs = np.zeros((5),dtype=np.uint8)
        for e in self.box.ravel().astype(int):
            if e >1:
                e = e-2
                #print (e)
                self.cntbxs[e] += 1

    def setBox(self,coords,idbox,box):
        lo,wo = coords #origin coords
        for w in range(box[0]):
            for l in range(box[1]):
                if w == 0 and l == 0:
                    self.box[lo,wo] = idbox
                else:
                    self.box[lo+l,wo+w] = 1
    
    def delBox(self,coords,box):
        lo,wo = coords #origin coords
        for w in range(box[0]):
            for l in range(box[1]):
                self.box[lo+l,wo+w] = 0

    def cutBox(self,bxs):
        boxpart = np.copy(self.box[:int(self.box.shape[0]/2),:])
        for l in range(boxpart.shape[0]):
            for w in range(boxpart.shape[1]):
                if boxpart[l,w] >1:
                    if self.itBoxCut((l,w),bxs.idxBox(boxpart[l,w]),boxpart):
                        self.delBox((l,w),bxs.idxBox(boxpart[l,w]))
        #self.printbox()
        return np.copy(self.box[:int(self.box.shape[0]/2),:]), np.copy(self.box[int(self.box.shape[0]/2):,:])

    def itBoxCut(self,coords,box,boxpart):
        lo,wo = coords #origin coords
        for w in range(box[0]):
            for l in range(box[1]):
                try:
                    if boxpart[lo+l,wo+w] != 0:
                        pass
                    else:
                        return True
                except:
                    return True
        return False


b2 = boxes2D()
b2.printbs()

ent = entity2D()
ent.solveFull(b2)
A1,A2 = ent.cutBox(b2)



def getPobIni(npob):
    pob = list()
    for i in range(npob):
        while True:
            #Generate object
            e = entity2D()
            #Solving full
            e.solveFull(b2)
            zeros = np.count_nonzero(e.cntbxs==0)
            #print(e.cntbxs,zeros)
            #I a box was not added compute a new entity
            if zeros == 0:
                break
        pob.append(e)
    return pob


def getFit(pob):
    zs = list()
    for i in pob:
        zs.append(i.unfilled)
    zs = np.array(zs,dtype=float)
    return zs

def getPond(ev):
    ev[np.where(ev==0)] = 0.5
    pond = 1/ev
    return pond/np.sum(pond)

def rulet(pond):
    npond = pond.shape[0]
    #Generar numero random
    r = np.random.random_sample()
    #Suma de la ruleta
    s = 0.0
    #Index para la seleccion
    idx = 0
    #Bucle de suma
    while True:
        s += pond[idx]
        if s>r:
            break
        else:
            idx += 1
            #Sentencia para retornar la ruleta
            if idx == npond:
                idx = 0
        #print("idx:",idx)
    return idx

def crossParents(p1,p2):
    global b2 
    part1,part2 = p1.cutBox(b2)
    part3,part4 = p2.cutBox(b2)

    #Mutation
    if np.random.randint(100) <= 10:
        rr = np.random.randint(2)
        if rr == 0:
            part1 = np.zeros(part1.shape,dtype= np.uint8)
        else:
            part2 = np.zeros(part1.shape,dtype= np.uint8)
    
    if np.random.randint(100) <= 10:
        rr = np.random.randint(2)
        if rr == 0:
            part3 = np.zeros(part1.shape, dtype=np.uint8)
        else:
            part4 = np.zeros(part1.shape, dtype=np.uint8)


    c1 = entity2D()
    c1.box = np.vstack([part1,part4])
    c1.solveFull(b2)
    zeros = np.count_nonzero(c1.cntbxs==0)
    if zeros != 0:
        c1 = None
    
    c2 = entity2D()
    c2.box = np.vstack([part3,part2])
    c2.solveFull(b2)
    zeros = np.count_nonzero(c2.cntbxs==0)
    if zeros != 0:
        c2 = None
    

    c3 = entity2D()
    r1 = np.random.randint(1,5)
    r2 = np.random.randint(1,5)
    c3.box = eval("np.vstack([part%d,part%d])" % (r1,r2))
    c3.solveFull(b2)
    zeros = np.count_nonzero(c3.cntbxs==0)
    if zeros != 0:
        c3 = None
    
    return c1,c2,c3


def selection(pob,pond):
    childs = 0
    newpob = list()
    holding = False

    while childs<100:
        idx = rulet(pond)
        if not holding:
            e1 = pob[idx]
            holding = True
        else:
            e2 = pob[idx]
            holding = False
            e3,e4,e5 = crossParents(e1,e2)
            if not e3 is None:
                childs += 1
                newpob.append(e3)
            if not e4 is None:
                childs += 1
                newpob.append(e4)
            if not e5 is None:
                childs += 1
                newpob.append(e5)

    finalpob = list()
    for e in pob:
        finalpob.append(e)
    for e in newpob:
        finalpob.append(e)
    return finalpob

def sortby(vals,n):
    vals = vals[vals[:,n].argsort()]
    return vals

def elits(pob,pond,n):
    pond = pond.reshape(-1,1)
    ind = np.arange(pond.shape[0]).reshape(-1,1)
    joint = np.hstack([pond,ind])

    joint = sortby(joint,0)
    joint = np.flip(joint,0)

    ind = joint[:,1]

    newpob = list()
    
    for i in range(n):
        newpob.append(pob[int(ind[i])])

    best = newpob[0]
    newpob2 = list()

    randd = np.arange(n,dtype=np.uint8)
    np.random.shuffle(randd)
    for i in range(n):
        newpob2.append(newpob[randd[i]])
    #print(len(newpob))

    return newpob2,best

npoblators = 200
pob = getPobIni(npoblators)
geners = 300
bests = list()
histo = np.empty((0,1))
plt.grid(True)
plt.title('Box problem')
plt.xlabel('Generations', fontsize=16)
plt.ylabel('Unfilled spaces', fontsize=16)

for g in range(geners):
    print("Gene: ",g)
    np.random.seed()
    #Evaluation
    print("Evaluating...")
    ev = getFit(pob)
    pond = getPond(ev)
    #Selection
    print("Making childs...")
    pob = selection(pob,pond)
    #Evaluation
    print("Evaluating...")
    ev = getFit(pob)
    pond = getPond(ev)
    #print(len(pob))
    #Elitism
    print("Elits...")
    pob,best = elits(pob,pond,npoblators)
    print(best.unfilled, best.cntbxs)
    histo = np.vstack([histo,best.unfilled])
    plt.scatter(g, histo[-1],c='blue')
    plt.pause(0.0000001)
    bests.append(best)

best.printbox()
plt.show()





