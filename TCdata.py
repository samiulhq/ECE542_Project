#   TCdata Class
import csv
import numpy as np
import matplotlib.pyplot as plt

class TCdata:
    data = []
    def __init__(self, f, n, challenge):
        self.challenge = challenge
        self.numGs = 0
        self.t = []
        self.numTPs = 0
        self.numPTs = 0
        self.numPos = 0
        self.numNeg = 0
        self.TC = self.loadData(f)
        self.network = self.loadNetwork(n)

# INTERNAL CLASS FUNCTIONS
    def loadData(self, f):
        # assumes time pt is first number in row remaining columns are gene data
        t = []
        firstRow = True
        firstPT = True
        if (self.challenge == 'D3'):
            lastTP = '200'
        if (self.challenge == 'D4'):
            lastTP = '1000'
        numGs = 0
        numTPs = 0
        numPTs = 0
        data = {}
        with open(f, 'r') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                if row[0]!='':
                    if firstRow:
                        numGs = len(row[1:])
                        Gs = row[1:]
                        data = dict((el, []) for el in Gs)
                        keys = list(data.keys())
                        currData = [[] for k in range(numGs)]
                        firstRow = False
                    else:
                        if firstPT:
                            t.append(int(row[0]))
                            numTPs = numTPs + 1
                            if (row[0]=='0'):
                                numPTs = numPTs + 1
                                currData = [[] for k in range(numGs)]
                                for i, v in enumerate(row[1:]):
                                    currData[i].append(float(v))
                            else:
                                if (row[0] != lastTP):
                                    for i, v in enumerate(row[1:]):
                                        currData[i].append(float(v))
                                else:
                                    firstPT = False
                                    for i, v in enumerate(row[1:]):
                                        currData[i].append(float(v))
                                    for i, v in enumerate(currData):
                                        data[keys[i]].append(v)
                        else:
                            if (row[0]=='0'):
                                numPTs = numPTs + 1
                                currData = [[] for k in range(numGs)]
                                for i, v in enumerate(row[1:]):
                                    currData[i].append(float(v))
                            else:
                                if (row[0] != lastTP):
                                    for i, v in enumerate(row[1:]):
                                        currData[i].append(float(v))
                                else:
                                    for i, v in enumerate(row[1:]):
                                        currData[i].append(float(v))
                                    for i, v in enumerate(currData):
                                        data[keys[i]].append(v)
        self.setTime(t)
        self.setGs(numGs)
        self.setPTs(numPTs)
        self.setTPs(numTPs)
        self.setTime(t)
        return data

    def loadNetwork(self,n):
        labels = {'+': 1, '-': 2}
        nwk = np.zeros((self.numGs,self.numGs),dtype='int')
        with open(n, 'r') as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                r = int(row[0].split("G")[1])-1
                c = int(row[1].split("G")[1])-1
                if row[2].isdigit():
                    l = row[2]
                else:
                    l = labels[row[2]]
                nwk[r, c] = l
            unique, counts = np.unique(nwk, return_counts=True)
            edgeCnts = dict(zip(unique, counts))
            self.setnumPos(edgeCnts[1])
            if len(edgeCnts) > 2:
                self.setnumNeg(edgeCnts[2])
        return nwk

    def setTime(self, t):
        self.t = t
        return

    def setTPs(self, TPs):
        self.numTPs = TPs
        return

    def setPTs(self, PTs):
        # numPTs = int(len(self.TC)/self.numTPs)
        self.numPTs = PTs
        return

    def setGs(self, G):
        self.numGs = G
        return

    def setnumPos(self, num):
        self.numPos = num
        return

    def setnumNeg(self, num):
        self.numNeg = num
        return

    def getTPstart(self, PT):
        return self.numTPs * (PT-1)

    def getTPstop(self, PT):
        start = self.getTPstart(PT)
        return (start + self.numTPs) - 1

    def getPT(self, keys, R, T, PT,dim):
        # initialize variables
        info = []
        timec = []
        label = 0
        rec = []
        # generate list of info
        info.append(keys[R])
        info.append(keys[T])
        info.append(PT+1)
        # create 1-D timecourse (numpy array)
        Rtc = np.array(self.TC[keys[R]][PT],dtype='float')
        Ttc = np.array(self.TC[keys[T]][PT],dtype='float')
        if dim ==1:
            timec = np.concatenate((Rtc, Ttc))
        else:
            timec = np.vstack((Rtc, Ttc))
        # timec.extend(self.TC[keys[T]][PT])
        # print('TC: {}'.format(timec))
        label = [0 if self.network[R][T]==0 else 1]
        rec = [info, timec, label]
        return rec


    def flipPT(self, keys, R, T, PT, dim):
        info = []
        timec = []
        label = 0
        rec = []
        info.append(keys[T])
        info.append(keys[R])
        info.append(PT+1)
        Rtc = np.array(self.TC[keys[R]][PT],dtype='float')
        Ttc = np.array(self.TC[keys[T]][PT],dtype='float')
        if dim == 1:
            timec = np.concatenate((Ttc, Rtc))
        else:
            timec = np.vstack((Rtc, Ttc))
        # timec.extend(self.TC[keys[T]][PT])
        # print('TC: {}'.format(timec))
        label = [0 if self.network[R][T]==0 else 2]
        rec = [info, timec, label]
        return rec



# EXTERNAL FUNCTIONS
    def get2TCwLabels(self, dim, bidir, PTs):
        # using network and TC output file for all gene pairs
        # dim = 1, then 1 x (#tps*2) vector created with label
        # dim = 3, then 2 x #tps vector created with label
        # bidir = False then only R-T recs created and labels are 0/1
        # bidir = True then R-T and T-R recs are created and labels are 0/1/2
        # PTs is a vector that indicates what perturbations (integers) to
        #   extract. if empty, then all PTs will be extracted
            # generate R-T and T-R recs labels = 0, 1, 2

        #   Get data from dataframe
        keys = list(self.TC.keys())
        data = []

        if (dim==1):
            if bidir:
                # print('Dim 1 T-R & R-T')
                for R in range(self.numGs):
                    for T in range(self.numGs):
                        if R!=T:
                            if PTs:
                                for PT in PTs:
                                    data.append(self.getPT(keys,R,T,PT-1,dim))
                                    data.append(self.flipPT(keys,R,T,PT-1,dim))
                            else:
                                for PT in range(self.numPTs):
                                    data.append(self.getPT(keys,R,T,PT,dim))
                                    data.append(self.flipPT(keys,R,T,PT,dim))
                return data
            else:
                # print('Dim 1 T-R only')
                for R in range(self.numGs):
                    for T in range(self.numGs):
                        if R!=T:
                            if PTs:
                                for PT in PTs:
                                    data.append(self.getPT(keys,R,T,PT-1,dim))
                            else:
                                for PT in range(self.numPTs):
                                    data.append(self.getPT(keys,R,T,PT,dim))
                return data

        elif (dim ==2):
            if bidir:
                # print('Dim 2 T-R & R-T')
                for R in range(self.numGs):
                    for T in range(self.numGs):
                        if R!=T:
                            if PTs:
                                for PT in PTs:
                                    data.append(self.getPT(keys,R,T,PT-1,dim))
                                    data.append(self.flipPT(keys,R,T,PT-1,dim))
                            else:
                                for PT in range(self.numPTs):
                                    data.append(self.getPT(keys,R,T,PT,dim))
                                    data.append(self.flipPT(keys,R,T,PT,dim))
            else:
                # print('Dim 2 T-R only')
                for R in range(self.numGs):
                    for T in range(self.numGs):
                        if R!=T:
                            if PTs:
                                for PT in PTs:
                                    data.append(self.getPT(keys,R,T,PT-1,dim))
                            else:
                                for PT in range(self.numPTs):
                                    data.append(self.getPT(keys,R,T,PT,dim))
            return data
        else:
            print("Invalid value for dim: choose only 1 or 2")

    def extractData(self, Genes):
        if not Genes:
            data = self.TC
        else:
            data = {x:self.TC[x] for x in Genes}

        return data

    def extractData2(self, PTs):
        # extractData2 returns a dictionary with 1 record / gene / PT
        # keys in dictionary are of the form G[gene #]-[perturbation #]
        data = self.extractData([])
        newdata = {}
        if not PTs:
            for k in data:
                for i in range(self.numPTs):
                    newkey = k + '-' + str(i+1)
                    newdata[newkey] = data[k][i]
        else:
            for k in data:
                for p in PTs:
                    newkey = k + '-' + str(p)
                    newdata[newkey] = data[k][p-1]
        return newdata



    def plotData(self, TC):
        leg = []
        for d in TC:
            tc = np.array(TC[d],dtype=float)
            for i, e in enumerate(tc):
                leg.append(d+'-P'+str(i+1))
                plt.plot(self.t,e)
        plt.title('Gene Expression Patterns')
        plt.xlabel('Time (units)')
        plt.ylabel('Normalized Expression')
        if len(leg) < 21:
            plt.legend(leg)
        plt.show()
