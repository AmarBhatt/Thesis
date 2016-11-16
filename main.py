from IPython.display import clear_output
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import History 
import numpy as np
import time
import matplotlib.pyplot as plt
from keras.callbacks import LambdaCallback
import random
from environment import *
from qnn import *


def testAlgo(init=0):
    i = 0
    print("Initial State:")
    if init==0:
        state = initGrid()
    elif init==1:
        state = initGridPlayer()
    elif init==2:
        state = initGridRand()    
    #print(dispGrid(state))
    status = 1
    #while game still in progress
    while(status == 1):
        qval = model.predict(state.reshape(1,64), batch_size=1)
        action = (np.argmax(qval)) #take action with highest Q-value
        print('Move #: %s; Taking action: %s' % (i, action))
        state = makeMove(state, action)
        #print(dispGrid(state))
        reward = getReward(state)
        if reward != -1:
            status = 0
            print("Reward: %s" % (reward,))
        i += 1 #If we're taking more than 10 actions, just stop, we probably can't win this game
        if (i > 10):
            print("Game lost; too many moves.")
            return 0,0,100
            break
    return 1,reward,i

def testBed(init,num):
    winloss = [];
    reward = [];
    steps = [];
    wins = 0;
    for i in range(num):
        print("*-*-*-*-* Test %s *-*-*-*-*"%(i,))
        g,r,s = testAlgo(init)
        winloss.append(g)
        reward.append(r)
        steps.append(s)
        wins+=g
    print("Wins/Game: %s/%s"%(wins,num))
    plt.figure(1)
    plt.subplot(211)
    plt.bar(range(num),winloss,1/1.5,color='blue')
    plt.title("Win/Loss per Test")
    plt.axis([0,num,0,1])
    plt.subplot(212)
    plt.plot(range(num), steps,'ro')
    plt.title("Steps per Test")
    plt.axis([0,num,0,10])
    plt.show()
    while(True):
        plt.pause(0.5)
    
        
def startTraining():
    epochs = 1000
    gamma = 0.9 #since it may take several moves to goal, making gamma high
    epsilon = 1
    for i in range(epochs):

        state = initGrid()
        status = 1
        #while game still in progress
        while(status == 1):
            #We are in state S
            #Let's run our Q function on S to get Q values for all possible actions
            qval = model.predict(state.reshape(1,64), batch_size=1)
            if (random.random() < epsilon): #choose random action
                action = np.random.randint(0,4)
            else: #choose best action from Q(s,a) values
                action = (np.argmax(qval))
            #Take action, observe new state S'
            new_state = makeMove(state, action)
            #Observe reward
            reward = getReward(new_state)
            #Get max_Q(S',a)
            newQ = model.predict(new_state.reshape(1,64), batch_size=1)
            maxQ = np.max(newQ)
            y = np.zeros((1,4))
            y[:] = qval[:]
            if reward == -1: #non-terminal state
                update = (reward + (gamma * maxQ))
            else: #terminal state
                update = reward
            y[0][action] = update #target output
            print("Game #: %s" % (i,))
            model.fit(state.reshape(1,64), y, batch_size=1, nb_epoch=1, verbose=1)
            state = new_state
            if reward != -1:
                status = 0
            clear_output(wait=True)
        if epsilon > 0.1:
            epsilon -= (1/epochs)   

def startTrainingWithExperienceReplay(stateType,epochs,gamma,epsilon,batchSize,buffer):
    replay = [] #stores tuples of (S, A, R, S')
    h = 0
    loss = [];
    timeTot = 0
    last_min = 0
    last_max = 2
    plt.figure(1)
    plt.ion()
    ax = plt.gca()
    ax.set_autoscale_on(True)
    line, = ax.plot([], [])
    plt.ylabel('error')
    plt.xlabel('iterations over time')
    plt.title('loss over time')
    
    for i in range(epochs):
        if stateType==0:
            state = initGrid()
        elif stateType==1:
            state = initGridPlayer()
        elif stateType==2:
            state = initGridRand()
        #state = initGridPlayer() #using the harder state initialization function
        status = 1
        print("Game %s/%s" % (i,epochs))
        #while game still in progress
        start = time.clock()
        while(status == 1):
            #We are in state S
            #Let's run our Q function on S to get Q values for all possible actions
            qval = model.predict(state.reshape(1,64), batch_size=1)
            if (random.random() < epsilon): #choose random action
                action = np.random.randint(0,4)
            else: #choose best action from Q(s,a) values
                action = (np.argmax(qval))
            #Take action, observe new state S'
            new_state = makeMove(state, action)
            #Observe reward
            reward = getReward(new_state)

            #Experience replay storage
            if (len(replay) < buffer): #if buffer not filled, add to it
                replay.append((state, action, reward, new_state))
            else: #if buffer full, overwrite old values
                if (h < (buffer-1)):
                    h += 1
                else:
                    h = 0
                replay[h] = (state, action, reward, new_state)
                #randomly sample our experience replay memory
                minibatch = random.sample(replay, batchSize)
                X_train = []
                y_train = []
                for memory in minibatch:
                    #Get max_Q(S',a)
                    old_state_m, action_m, reward_m, new_state_m = memory
                    old_qval = model.predict(old_state_m.reshape(1,64), batch_size=1)
                    newQ = model.predict(new_state_m.reshape(1,64), batch_size=1)
                    maxQ = np.max(newQ)
                    y = np.zeros((1,4))
                    y[:] = old_qval[:]
                    if reward_m == -1: #non-terminal state
                        update = (reward_m + (gamma * maxQ))
                    else: #terminal state
                        update = reward_m
                    y[0][action_m] = update
                    X_train.append(old_state_m.reshape(64,))
                    y_train.append(y.reshape(4,))

                X_train = np.array(X_train)
                y_train = np.array(y_train)
                hist = model.fit(X_train, y_train, batch_size=batchSize, nb_epoch=1, verbose=0)
                #Plot loss function
                loss.append(hist.history['loss'][0])
                line.set_xdata(np.append(line.get_xdata(), len(loss)-1))
                line.set_ydata(np.append(line.get_ydata(), loss[-1]))
                ax.relim()
                ax.autoscale_view(True,True,True)
                plt.draw()
                plt.pause(0.000000000000000001)

                state = new_state
            if reward != -1: #if reached terminal state, update game status
                status = 0
                end = time.clock()
                timeTot +=(end-start)
                #annotate graph
                #if(i%50):
                    #ax.annotate(str(i), ((len(loss)-1),last_max))
                    #ax.axvline(x=(len(loss)-1),linewidth=1, color='r')
                #if(i%100):
                    #last_max = max(loss[last_min:])-0.01
                    #ax.set_ylim(0,last_max)
                    #last_min = len(loss)-1
            clear_output(wait=True)
        if epsilon > 0.1: #decrement epsilon over time
            epsilon -= (1/epochs)
        print("Time Elapsed: %f"%(end-start,))
        print("Avg. Time/Game: %f"%(timeTot/(i+1),))
        print("Loss: %f"%(loss[-1],))
    return loss   




#state = en.initGridRand()

model = createModel()

#print(model.predict(state.reshape(1,64), batch_size=1))
#just to show an example output; read outputs left to right: up/down/left/right

epochs = 3000 #number of games
gamma = 0.975 #discount factor
epsilon = 1 #policy
batchSize = 40 #mini-batch ammount
buffer = 80 #experience replay size
stateType = 1; #0 - stationary, 1- random player, 2-random environment
#startTraining()
#testAlgo(init=0)
#testAlgo(init=1)
#testAlgo(init=1)
#testAlgo(init=1)
#model.compile(loss='mse', optimizer=rms)#reset weights

#plot_loss_callback = LambdaCallback(on_epoch_end=lambda epoch, logs: plt.plot(np.arange(epoch), logs['loss']))

#history = History()

loss = startTrainingWithExperienceReplay(2,10000,gamma,epsilon,batchSize,buffer)

#print(loss)

testBed(2,1000)

#plt.figure(figsize=(6, 3))
#plt.plot(range(len(loss)),loss)

#plt.show()


#print("*-*-*-*-* Test 1 *-*-*-*-*")
#testAlgo(init=1)
#print("*-*-*-*-* Test 2 *-*-*-*-*")
#testAlgo(init=1)
#print("*-*-*-*-* Test 3 *-*-*-*-*")
#testAlgo(init=1)
#print("*-*-*-*-* Test 4 *-*-*-*-*")
#testAlgo(init=1)
