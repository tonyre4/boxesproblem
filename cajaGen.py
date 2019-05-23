#Dimensions
#Big Box: w:30 l:50 h:25
#Boxes: 
    #1: 1x2x1
    #2: 2x2x1
    #3: 2x3x1
    #4: 5x4x2
    #5: 5x1x2

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
        self.boxsmat = list()
        self.boxcntr = 0

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
                        self.setBox((l,w),idx,rndbox,self.box)

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

    def setBox(self,coords,idbox,box,container):
        boxfts = list()
        boxcoords = np.empty((0,2),dtype=np.uint8)
        lo,wo = coords #origin coords
        for w in range(box[0]):
            for l in range(box[1]):
                if w == 0 and l == 0:
                    container[lo,wo] = idbox
                    boxfts.append([self.boxcntr,idbox])
                    #self.boxcntr += 1
                    boxcoords = np.vstack([boxcoords,[lo,wo]])
                else:
                    container[lo+l,wo+w] = 1
                    boxcoords = np.vstack([boxcoords,[lo+l,wo+w]])
        boxfts.append(boxcoords)
        #self.boxsmat.append(boxfts)
        #print(boxfts)
        #print(self.boxsmat)
        #self.printbox()

    def delBox(self,coords,box):
        lo,wo = coords #origin coords
        for w in range(box[0]):
            for l in range(box[1]):
                self.box[lo+l,wo+w] = 0

    def cutBox(self,bxs,height):
        boxpart = np.copy(self.box[:int(self.box.shape[0]/2),:])
        newbox1 = np.zeros(self.box.shape, dtype=np.uint8)
        newbox2 = np.zeros(self.box.shape, dtype=np.uint8)
        for l in range(self.box.shape[0]):
            for w in range(self.box.shape[1]):
                if self.box[l,w] >1:
                    idd = self.box[l,w]
                    if l<height:
                        boxtofill = newbox1
                    else:
                        boxtofill = newbox2
                    self.setBox([l,w],idd,bxs.idxBox(idd),boxtofill)
        return newbox1,newbox2

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

pob = getPobIni(100)

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

def crossMatrix(p1,p2,p3,p4):
    newmat1 = p1
    newmat2 = p3

    return newmat1,newmat2


def crossParents(p1,p2):
    global b2 
    h = np.random.randint(1,p1.box.shape[0])
    part1,part2 = p1.cutBox(b2,h)
    part3,part4 = p2.cutBox(b2,h)
    c1 = entity2D()
    c2 = entity2D()

    c1.box,c2.box = crossMatrix(part1,part2,part3,part4)

    #Mutation
    if np.random.randint(100) <= 10:
        c2.solveFull(b2)
    if np.random.randint(100) <= 10:
        c2.solveFull(b2)


    zeros = np.count_nonzero(c1.cntbxs==0)
    if zeros != 0:
        c1 = None
    if zeros != 0:
        c2 = None

    return c1,c2


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
            e3,e4 = crossParents(e1,e2)
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

geners = 300
bests = list()
for g in range(geners):
    np.random.seed()
    #Evaluation
    ev = getFit(pob)
    pond = getPond(ev)
    #Selection
    pob = selection(pob,pond)
    #Evaluation
    ev = getFit(pob)
    pond = getPond(ev)
    #print(len(pob))
    #Elitism
    pob,best = elits(pob,pond,100)
    print("Gene: ",g)
    print(best.unfilled, best.cntbxs)
    bests.append(best)

best.printbox()





