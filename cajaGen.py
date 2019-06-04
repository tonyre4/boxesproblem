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

    def mergeBoxes(self,bx1,bx2,bxs):
        self.box = bx1 #Setting first part of cutted box
        #print(np.flip(bx2,0),"\n")
        #print(np.flip(bx1,0),"\n")

        for l in range(self.box.shape[0]):
            for w in range(self.box.shape[1]):
                if self.box[l,w] == 0 and bx2[l,w]>1:
                    idbox = bx2[l,w]
                    if self.itBoxFit((l,w),bxs.idxBox(idbox)):
                        self.setBox((l,w),idbox,bxs.idxBox(idbox))
        #self.printbox()
        #exit()
        self.getFeatures()

    def setBox(self,coords,idbox,box,container=None):
        if container is None:
            container = self.box
        lo,wo = coords #origin coords
        for w in range(box[0]):
            for l in range(box[1]):
                if w == 0 and l == 0:
                    container[lo,wo] = idbox
                else:
                    container[lo+l,wo+w] = 1
    
    def delBox(self,coords,box):
        lo,wo = coords #origin coords
        for w in range(box[0]):
            for l in range(box[1]):
                self.box[lo+l,wo+w] = 0

    def cutBox(self,cut,bxs):
        p1 = np.zeros(self.box.shape)
        p2 = np.zeros(self.box.shape)

        for l in range(self.box.shape[0]):
            for w in range(self.box.shape[1]):
                idbox = self.box[l,w]
                if idbox >1:
                    if l<=cut:
                        self.setBox((l,w),idbox,bxs.idxBox(idbox),container=p1)
                    else:
                        self.setBox((l,w),idbox,bxs.idxBox(idbox),container=p2)
        p1 = p1.astype(np.uint8)
        p2 = p2.astype(np.uint8)
        return p1,p2

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

def mutation(c,mutRate,b2):
    if np.random.randint(0,100) <= mutRate:
        cntr = 0
        p = False
        while True:
            if cntr<50:
                #if p:
                #print(cntr)
                current = np.copy(c.box)
                c.solveFull(b2)
                if np.any(current != c.box):
                    #print(np.flip(current,0),"\n")
                    #c1.printbox()
                    #exit()
                    #print("Child mutated!")
                    break
                cntr += 1
            else:
                #print("Deleting Boxes")
                #c.printbox()
                #print("Stuck Child :/")
                bxtodel = np.random.randint(1,10)
                #print("deleting %d boxes from child from random locations" % (bxtodel))
                for i in range(bxtodel):
                    while True:
                        l = np.random.randint(0,c.box.shape[0])
                        w = np.random.randint(0,c.box.shape[1])
                        if c.box[l,w]>1:
                            c.delBox((l,w),b2.idxBox(c.box[l,w]))
                            cntr = 0
                            p = True
                            break
                #print("\n")
                #c.printbox()
                #exit()

def crossParents(p1,p2,mutRate):
    global b2 

    cut = np.random.randint(1,p1.box.shape[0]-2)
    part1,part2 = p1.cutBox(cut,b2)
    part3,part4 = p2.cutBox(cut,b2)

    c1 = entity2D()
    c1.mergeBoxes(part1,part4,b2)
    c2 = entity2D()
    c2.mergeBoxes(part3,part2,b2)

    #Mutation
    mutation(c1,mutRate,b2)
    mutation(c2,mutRate,b2)
    

    zeros = np.count_nonzero(c1.cntbxs==0)
    if zeros != 0:
        c1 = None

    zeros = np.count_nonzero(c2.cntbxs==0)
    if zeros != 0:
        c2 = None

    return c1,c2


def selection(nchilds,pob,pond):
    childs = 0
    newpob = list()
    holding = False

    while childs<nchilds:
        idx = rulet(pond)
        if not holding:
            e1 = pob[idx]
            holding = True
        else:
            e2 = pob[idx]
            holding = False
            e3,e4 = crossParents(e1,e2,10)
            if not e3 is None:
                childs += 1
                newpob.append(e3)
            if not e4 is None:
                childs += 1
                newpob.append(e4)

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

b2 = boxes2D() #Boxes object
npoblators = 300
newchildsn = 400
pob = getPobIni(npoblators)
geners = 1500
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
    #print("Evaluating...")
    ev = getFit(pob)
    pond = getPond(ev)
    #Selection
    #print("Making childs...")
    pob = selection(newchildsn,pob,pond)
    #Evaluation
    #print("Evaluating...")
    ev = getFit(pob)
    pond = getPond(ev)
    #print(len(pob))
    #Elitism
    #print("Elits...")
    pob,best = elits(pob,pond,npoblators)
    print(best.unfilled, best.cntbxs)
    histo = np.vstack([histo,best.unfilled])
    plt.scatter(g, histo[-1],c='blue')
    plt.pause(0.0000001)
    bests.append(best)

best.printbox()
plt.show()





