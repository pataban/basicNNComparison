from constants import *
from dataCreation import *
from Layer import Layer


class Network():
    def __init__(self):
        self.layers=[]
        self.layers.append(Layer(NEURON_COUNT[0],NEURON_COUNT[1],Layer.reLU,Layer.reLUDerivative))
        #self.layers.append(Layer(NEURON_COUNT[1],NEURON_COUNT[2],Layer.reLU,Layer.reLUDerivative))
        #self.layers.append(Layer(NEURON_COUNT[2],NEURON_COUNT[3],Layer.reLU,Layer.reLUDerivative))
        self.layers.append(Layer(NEURON_COUNT[1],NEURON_COUNT[2],Layer.softMax))


    def fit(self,xTrain,yTrain,xTest,yTest):
        yTrain=Network.yAsSoftmax(yTrain)
        #xTrain=list(xTrain)
        #yTrain=list(yTrain)

        epoch=1
        successRate=0.0
        lastSuccessRate=-1.0
        while(successRate<SUCCESS_MIN and successRate-lastSuccessRate>SUCCESS_EARLY_STOP):
            """self.fitBatch(np.asarray(random.sample(xTrain,BATCH_SIZE)),
                    np.asarray(random.sample(yTrain,BATCH_SIZE)))"""
            for i in range(0,xTrain.shape[0],BATCH_SIZE):
                #if i%100==0:
                    #print(".", end='')
                self.fitBatch(xTrain[i:i+BATCH_SIZE],yTrain[i:i+BATCH_SIZE])
            #print("")

            lastSuccessRate=successRate
            successRate=self.test(xTest[0:TEST_SIZE],yTest[0:TEST_SIZE])
            print("epoch:", epoch, "->", successRate)
            epoch+=1

        """trainCount=100
        for i in range(0,trainCount):
            if BATCH_SIZE*i%1000==0:
                print(".")
            self.fitBatch(xTrain[i*BATCH_SIZE:(i+1)*BATCH_SIZE],Network.yAsSoftmax(yTrain[i*BATCH_SIZE:(i+1)*BATCH_SIZE]))"""

    
    def fitBatch(self,x,y):
        self.expand()
        a=[x]
        z=[]
        for layer in self.layers:
            z.append(np.matmul(layer.w,a[-1]) + layer.b)
            a.append(layer.aFun(z[-1]))

        grads=list(range(0,len(self.layers)))       #empty list of len == len(layers)
        grads[-1]= -(y-z[-1])  #a[-2].T             #10x1
        for i in reversed(range(0,len(self.layers)-1)):
            #print(self.layers[i+1].w.T.shape)                   #150x10
            #print(grads[i+1].shape)                             #10x1
            #print(self.layers[i].activateFunDer(z[i]).shape )   #150x1
            #print(a[i].T.shape)                                 #1x300
            grads[i] = np.matmul(self.layers[i+1].w.transpose((0,2,1)),  grads[i+1]) * self.layers[i].activateFunDer(z[i])  #a[i].T

        for layer, grad, ai in zip(self.layers,grads,a):
            layer.w = layer.w[0] - (LEARNING_SPEED / x.shape[0]) * np.matmul(grad,ai.transpose((0,2,1))).sum(axis=0)
        for layer, grad in zip(self.layers,grads):
            layer.b = layer.b[0] - LEARNING_SPEED / x.shape[0] * grad.sum(axis=0)


    def classify(self,x):
        for layer in self.layers:
            x=layer.calc(x)
        return x.argmax(axis=1)


    def test(self,x,y):
        classes=self.classify(x).flatten()
        res=(classes-y).astype(bool).astype(int)
        
        #print("y\n",y)
        #print("class\n",classes.T)
        #print("res\n",res)

        successRate=(x.shape[0]-res.sum())/x.shape[0]
        #print("result",successRate)
        return successRate


    def expand(self):
        for layer in self.layers:
            layer.w=np.stack([layer.w for _ in range(BATCH_SIZE)],axis=0)
            layer.b=np.stack([layer.b for _ in range(BATCH_SIZE)],axis=0)
    

    def yAsSoftmax(y):
        ySM=np.full((y.shape[0],10,1),fill_value=0)        
        for yi, iSM in zip(y, range(0,ySM.shape[0])):
            ySM[iSM,yi]=1
        return ySM
