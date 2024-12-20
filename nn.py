
from typing import List
import numpy as np
from multiprocessing import Process, Lock

class NueralNetwork():

    def __init__(self, layout, activation="sigmoid",**kwargs) -> None:
        self.layerLens = layout
        self.weights = []
        self.biases = []
        self.activateLast = False

        if kwargs.get("activateLast",False):
            self.activateLast = True
        if activation == "sigmoid":
            self.activation = np.vectorize(self.sigmoid)
            self.activationD = np.vectorize(self.sigmoidD)
        elif activation == "relu":
            self.activation = np.vectorize(self.relu)
            self.activationD = np.vectorize(self.reluD)
        elif activation == "lrelu":
            self.activation = np.vectorize(self.lrelu)
            self.activationD = np.vectorize(self.lreluD)
        elif activation == "tanh":
            self.activation = np.vectorize(self.tanh)
            self.activationD = np.vectorize(self.tanhD)

        for i in range(len(self.layerLens)):
            if i != len(self.layerLens) - 1:
                self.weights.append(np.random.randn(self.layerLens[i+1],self.layerLens[i])/(self.layerLens[i+1]+self.layerLens[i]))
                self.biases.append(np.zeros(self.layerLens[i+1]).reshape(-1,1))


    def sigmoid(self,x):
        return 1/(1+np.e**-x)
    def sigmoidD(self,x):
        return x * (1 - x)
    
    def tanh(self,x):
        return np.tanh(x)
    def tanhD(self,x):
        return 1 - x**2
    
    def relu(self,x):
        return max(0,x)
    def reluD(self,x):
        if x <= 0:
            return 0
        return 1
    
    def lrelu(self,x):
        return max(x,0.01*x)
    def lreluD(self,x):
        if x <= 0:
            return 1
        return 0.01
    
    def propegateFull(self,input:np.ndarray,layer=-1):
        if(layer == -1):
            layer = len(self.layerLens) - 1

        for i in range(layer):
            # print(i)
            assert input.shape[0] == self.weights[i].shape[1], f"Shape mismatch {self.weights[i].shape} @ {input.shape}"
            input = self.weights[i] @ input + self.biases[i]
            if i != len(self.layerLens)-2 or self.activateLast:
                input = self.activation(input)
        return input
    
    def getActivations(self,input:np.ndarray):
        Z = [input]
        for i in range(len(self.layerLens)-1):
            input = self.weights[i] @ input + self.biases[i]
            if i != len(self.layerLens)-2 or self.activateLast:
                input = self.activation(input)
            Z.append(input)
        return Z

    def back_propegate(self,input:np.ndarray,target:np.ndarray):
        Z = self.getActivations(input)
        derror = (Z[-1] - target)
        loss = sum((derror**2)/(2))/derror.shape[0]
        # print(loss)

        if self.activateLast:
            startingD = derror * self.activationD(Z[-1])
        else:
            startingD = derror

        delta = [startingD]
        # zshaped = np.empty((self.layerLens[-1],self.layerLens[-2]))
        # for idx,_ in np.ndenumerate(zshaped):
        #         zshaped[idx[0],idx[1]] = Z[-2][idx[1]]

        deltaW = [startingD * Z[-2].T]
        for i in range(1,len(self.layerLens)-1):
            layerActivation = Z[-(i+2)]
            # zshaped = np.empty((self.layerLens[-(i+1)],self.layerLens[-(i+2)]))
            # for idx,_ in np.ndenumerate(zshaped):
            #         zshaped[idx[0],idx[1]] = layerActivation[idx[1]]

            # newDelta =  (self.weights[-i].T @ delta[-1]) * self.activationD(Z[-(i+1)])
            newDelta =  (delta[-1].T @ self.weights[-i]).T * self.activationD(Z[-(i+1)])

            wieghtsDelta = newDelta @ layerActivation.T
            delta.append(newDelta)
            deltaW.append(wieghtsDelta)

        return loss, delta, deltaW
    
    def train(self,x,y,mutex,learningRate,length):
        newWeights = [np.zeros(w.shape) for w in self.weights]
        newBiases = [np.zeros(b.shape) for b in self.biases]
        loss, delta, deltaW = self.back_propegate(x,y)
        for i in range(len(self.layerLens)-1):
            assert newWeights[i].shape == deltaW[-(i+1)].shape, f"shape mismatch {deltaW[-(i+1)].shape} @ {newWeights[i].shape}"
            newWeights[i] = newWeights[i] + deltaW[-(i+1)]
            newBiases[i] = newBiases[i] + delta[-(i+1)]

        for i in range(len(self.weights)):
            mutex.acquire()            
            self.batchLoss += loss
            self.weights[i] = self.weights[i] - (learningRate)/length*newWeights[i] 
            self.biases[i] = self.biases[i] - (learningRate)/length*newBiases[i] 
            mutex.release()

    def mini_batch_t(self,X,Y,learningRate):

        self.batchLoss = 0


        
        processes = []
        mutex = Lock()
        for id,xy in enumerate(zip(X,Y)):
            x = xy[0]
            y = xy[1]
            process = Process(name=f"{id}",target=self.train,args=(x,y,mutex,learningRate,len(X)))
            process.start()
            processes.append(process)
            pass
        for process in processes:
            process.join()
            print(f"Joined process {process.name}")
        return self.batchLoss, None

    def mini_batch(self,X,Y,learningRate):
        newWeights = [np.zeros(w.shape) for w in self.weights]
        newBiases = [np.zeros(b.shape) for b in self.biases]
        batchLoss = 0
        for x,y in zip(X,Y):
            loss, delta, deltaW = self.back_propegate(x,y)
            batchLoss += loss
            for i in range(len(self.layerLens)-1):
                # assert newWeights[i].shape == deltaW[-(i+1)].shape, f"shape mismatch {deltaW[-(i+1)].shape} @ {newWeights[i].shape}"
                newWeights[i] = newWeights[i] + deltaW[-(i+1)]
                newBiases[i] = newBiases[i] + delta[-(i+1)]

            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] - (learningRate)/len(X)*newWeights[i] 
                self.biases[i] = self.biases[i] - (learningRate)/len(X)*newBiases[i] 
        
        return batchLoss/len(X), newWeights
        

    def batch(self, X, Y, size, learningRate,threaded=True):
        batches = len(X) // size
        mb = None
        if(threaded):
            mb = self.mini_batch_t
        else:
            mb = self.mini_batch
        # Process full batches
        batchLoss = 0
        for i in range(batches):
            # Slice X and Y for the current batch
            X_batch = X[size * i:size * (i + 1)]
            Y_batch = Y[size * i:size * (i + 1)]
            # print("Processing Batch:", i)
            loss, _ = mb(X_batch, Y_batch, learningRate)
            batchLoss += loss
        # Process remaining samples (if any)
        if len(X) % size != 0:
            X_remaining = X[batches * size:]
            Y_remaining = Y[batches * size:]
            # print("Processing Remaining Samples")
            loss, _ = mb(X_remaining, Y_remaining, learningRate)
            batchLoss += loss
        return loss/batches


        
if __name__ == '__main__':
    import signal
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import normalize
    

    heartdf = pd.read_csv('HeartDisease (1).csv')
    print(heartdf.isna().sum())
    heartdf = heartdf.drop("ExerciseAngina", axis=1)
    heartdf = heartdf.dropna()

    heartdf.loc[:,"ST_Slope"] = (heartdf["ST_Slope"] == "Flat").astype(int)
    heartdf.loc[:,"Sex"] = (heartdf["Sex"] == "M").astype(int)


    df = pd.get_dummies(heartdf, columns=["ChestPainType","RestingECG"])

    numberCols= df.select_dtypes(include=['number']).columns

    df[numberCols] = (df[numberCols] - df[numberCols].min()) / (df[numberCols].max() - df[numberCols].min())

    # print(df)
    df = df.astype(int)
    X = df.drop("HeartDisease", axis=1)
    Y = pd.DataFrame(df["HeartDisease"],columns=["HeartDisease"])
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.8,random_state=69)
    Xtrain = [x.reshape(-1,1) for x in Xtrain.to_numpy()]
    Ytrain = [y.reshape(-1,1) for y in Ytrain.to_numpy()]
    Xtest = [x.reshape(-1,1) for x in Xtest.to_numpy()]
    Ytest = [y.reshape(-1,1) for y in Ytest.to_numpy()]
    # print(Ytrain) 
    stop = False
    def handle_sigint(signal_number, frame):
        global stop
        stop = True

    nn = NueralNetwork([1,20,20,1],"tanh",activateLast=False)

    examples = 5000
    sampleInput = []
    
    for i in range(examples):
        sampleInput.append(np.random.rand(nn.layerLens[0],1))
    # sampleInput = normalize(sampleInput)

    def norm(inp,a=0,b=1):
        minVal = min(inp)
        maxd = max(inp)
        return [((x - minVal)/(maxd - minVal)) * (a-b) + a for x in inp]
    

    sampleInput = [3*((np.random.rand(nn.layerLens[0], 1))) for _ in range(examples)]
    sampleOutput = [np.floor(inp) for inp in sampleInput]
    a = -1
    b = 1
    minValI, maxValI = min(sampleInput)[0][0], max(sampleInput)[0][0]
    minValO, maxValO = min(sampleOutput)[0][0], max(sampleOutput)[0][0]
    sampleInput = norm(sampleInput)
    sampleOutput = norm(sampleOutput)


    nn = NueralNetwork([1,20,20,1],"tanh",activateLast=False)

    output = nn.propegateFull(sampleInput[0])
    epochs = 100
    loss = 10000
    epoch = 0
    decay = 0
    
    while epoch < epochs and not stop:
        learningRate = 0.01*(1/(1+decay*epoch))
        loss = nn.batch(sampleInput,sampleOutput,100,learningRate,True)
        if epoch % ((epochs//10) + 1) == 0:
                    print(f"Epoch: {epoch}, loss={loss}, learningRate = {learningRate}")
        epoch += 1
    exit()
    from sklearn.metrics import accuracy_score


    def propegateFloat(x):
        # print("X: ",x, flush=True)
        print(f"x: {x}")
        x = (x - minValI)/(maxValI - minValI) 
        print(x)
        res = nn.propegateFull(np.array([[x]]))

        res = (res-a)*(maxValO - minValO)/(b-a) + minValO
        return res

    propegate = np.vectorize(propegateFloat)

    # print([nn.propegateFull(res) for res in Xtest])
    # resultT = [(1 if float(nn.propegateFull(res)[0][0]) > 0.5 else 0) for res in Xtrain]
    # result = [(1 if float(nn.propegateFull(res)[0][0]) > 0.5 else 0) for res in Xtest]
    # Xtraining = [(res[0][0]) for res in Xtrain]

    # Ytrain =  [int((res)[0][0]) for res in Ytrain]
    # Ytest =  [int((res)[0][0]) for res in Ytest]

    # print("RESULT",result[:10])
    # print("YTest",Ytest[:10])

    # print(accuracy_score((Ytrain),(resultT)),accuracy_score(Ytest,result))
    # print(nn.biases)

    x = np.arange(0,10,0.1)
    y = propegate(x)
    import matplotlib.pyplot as plt

    plt.plot(x,y)
    # plt.scatter(sampleInput,sampleOutput)
    plt.plot(x,np.floor(x))
    plt.ylabel('some numbers')
    plt.show()



