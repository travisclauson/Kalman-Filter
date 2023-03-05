import numpy as np
from math import *
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

t = 5
currTime = 0
useOldValues = False

A = np.array([[1,1],[0,1]])
B = np.array([[1/2],[1]])
G = np.array([[1/2],[1]])
C = np.array([[1,0]])
wind_variance = 1
R = np.matmul(G,np.transpose(G))*wind_variance # var*GG^T
Q = 8 #sensor variance
useSensors = False
plotEllipses = True

x0 = 0  # intial position 
v0 = 0  # intial velocity
X0 = np.array([[x0],[v0]])  # initial state vector
realX0 = X0  # initial state vector
u1 = np.array([[0]])  # initial control (wind) vector
sigma0 = np.array([[0,0],[0,0]]) # initial covariance matrix
xTotal = np.array(x0)  # history of position estimation
vTotal = np.array(v0)  # history of velocity estimation
rxTotal = np.array(0)  # history of real position
rvTotal = np.array(0)  # history of real velocity
zTotal = np.array(0)  # history of measurements
wTotal = np.array(0)  # history of wind
sigmaTotal = [np.array(sigma0)] # history of covariance matrix
positionErrors = np.empty(0)

fig, ax = plt.subplots()
ax.set_xlim(-10,10)
ax.set_ylim(-5,5)
plt.ylabel('Velocity')
plt.xlabel('Position')
plt.title('Uncertainty Ellipses for t=1 to t=5')
lineColors = ['b','g','r','c','m','y']


def store_variables(X1, realX1, z1, u1, w1, sigma1, currTime):
    global xTotal, vTotal, rxTotal, rvTotal, zTotal, uTotal, wTotal, sigmaTotal
    # Overwrite the data if the current time has been seen before (wind values are not overwritten)
    if useOldValues:
        xTotal[currTime] = X1[0]
        vTotal[currTime] = X1[1]
        #rxTotal[currTime] = realX1[0]
        #rvTotal[currTime] = realX1[1]
        #uTotal[currTime] = u1
        #sigmaTotal[currTime] = sigma1
    # Append the data if the current time has not been seen before
    else:
        xTotal = np.append(xTotal,X1[0])
        vTotal = np.append(vTotal,X1[1])
        rxTotal = np.append(rxTotal,realX1[0])
        rvTotal = np.append(rvTotal,realX1[1])
        #uTotal = np.append(uTotal,u1)
        wTotal = np.append(wTotal,w1)
        zTotal = np.append(zTotal,z1)
        #sigmaTotal = np.append(sigmaTotal,sigma1)

def plot_uncertainty_ellipse(x, y, C, currTime):
    # Compute eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    # Compute the angle of the ellipse
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    # Scale the eigenvectors by the square root of the corresponding eigenvalues
    semi_axes = np.sqrt(eigenvalues) * 2.
    # Plot the uncertainty ellipse
    return(Ellipse(xy=(x,y), width=semi_axes[0], height=semi_axes[1], angle=angle, lw=.5, facecolor='none', label = f"t={currTime}", edgecolor=lineColors[currTime]))


def kalmanAlgorithm(X0, sigma0, u, z, currTime=currTime, pGPS = 1.0):
    # Estimate State always
    X1 = np.matmul(A,X0) + np.matmul(B,u1) # AX + BU (state transition)
    sigma1 = np.matmul(np.matmul(A,sigma0),np.transpose(A)) + R # A*Sigma*A^T+R
    
    takeSensorReading = np.random.binomial(1,pGPS) # 1 if we take a sensor reading, 0 if we don't

    if (currTime == 5): #Problem 2.2
        print(f"\nTime t=5 Actual State Values: {realX0}")
        print("\nBEFORE SENSOR READING:")
        print(f"State Esitmation: {X1}")
        print(f"Covariance Matrix: {sigma1}")
        z = 10

    if (takeSensorReading or currTime == 5): # If we have a sensor reading
        # Kalman Gain
        K = np.matmul(np.matmul(sigma1,np.transpose(C)),np.linalg.inv(np.matmul(np.matmul(C,sigma1),np.transpose(C))+Q)) # Sigma*C^T*(C*Sigma*C^T+Q)^-1
        
        # Update State
        X1 = X1 + np.matmul(K,(z-np.matmul(C,X1))) # X + K*(z-C*X)
        sigma1 = np.matmul((np.identity(2)-np.matmul(K,C)),sigma1) # (I-K*C)*Sigma
        if (currTime == 5):
            print("\nAFTER SENSOR READING: z=10")
            print(f"New State Esitmation: {X1}")
            print(f"New Covariance Matrix: {sigma1}")

    return X1, sigma1


def iterate_simulation(realX0, currTime): # Runs the simulation for the given time step, 
    if not useOldValues: # If the current time has not been seen before
        w1 = np.random.normal(0,1)  # W = random wind amount
        realX1 = np.matmul(A,realX0) + np.matmul(B,u1) + w1*G # AX + BU (state transition)
    else: 
        w1 = wTotal[currTime] # Use the wind value stored in the history
        realX1 = np.array([[rxTotal[currTime]], [rvTotal[currTime]]]) # Use the state value stored in the history
    return realX1, w1

def runFilter(start, end, X0=X0, sigma0=sigma0, pGPS = 1.0): # Runs the filter for the given time range, overwrites old time steps
    global realX0, useOldValues
    for i in range(start,end+1):
        currTime = i
        
        try: dataPoints = len(xTotal)
        except: dataPoints = 0
        if currTime >= dataPoints: useOldValues = False # If the current time has not been seen before
        else: useOldValues = True # If the current time has been seen before
       
        # Real (Grounded) System
        realX1, w1 = iterate_simulation(realX0, currTime)

        # Sensor Reading
        z1 = np.matmul(C,realX1) + np.random.normal(0,np.sqrt(Q)) # Real Altitude + Sensor Noise

        # Kalman Filter
        X1, sigma1 = kalmanAlgorithm(X0, sigma0, u1, z1, currTime, pGPS)

        # store variables for plotting
        store_variables(X1, realX1, z1, u1, w1, sigma1, currTime)

        # update variables as time passes
        sigma0 = sigma1
        X0 = X1
        realX0 = realX1

        # Plot uncerrtainty ellipse
        if plotEllipses:
            ellipse = plot_uncertainty_ellipse(X1[0], X1[1], sigma1, currTime)
            ax.add_artist(ellipse)
        positionError = np.subtract(xTotal[-1],rxTotal[-1])
    return(positionError, X1, sigma1)


# Run the filter for t=1 to t=5
positionError,X0,sigma0 = runFilter(1,5, X0, sigma0, pGPS = 0.0)
leg = plt.legend(loc='best')
plotEllipses = False


# ----------------------- PROBLEM 2.3 ----------------------- #
# Save the state at t=5
recallStateEstimation = X0
recallSigmaEstimation = sigma0
print(f"State at t=5: {recallStateEstimation}")
print(f"Sigma at t=5: {recallSigmaEstimation}")

# Run the filter for t=5 to t=20
positionErrors = np.append(positionErrors, runFilter(6,20, recallStateEstimation, recallSigmaEstimation, pGPS = .1)[0])
print(f"\npGPS=.9  GROSS ERROR at t=20: {positionErrors[-1]}")
# Run the filter for t=5 to t=20
positionErrors = np.append(positionErrors, runFilter(6,20, recallStateEstimation, recallSigmaEstimation, pGPS = .5)[0])
print(f"\npGPS=.5  GROSS ERROR at t=20: {positionErrors[-1]}")
# Run the filter for t=5 to t=20
positionErrors = np.append(positionErrors, runFilter(6,20, recallStateEstimation, recallSigmaEstimation, pGPS = .9)[0])
print(f"\npGPS=.1  GROSS ERROR at t=20: {positionErrors[-1]}")
probabilityGPS = [.1, .5, .9]
positionErrors = np.absolute(positionErrors)

# ---------------------- PROBLEM 3.1 ---------------------- #
X0 = np.array([[5.0],[1.0]])  # initial state vector
realX0 = X0
u1 = np.array([[1.0]])
sigma0 = np.array([[0.0,0.0],[0.0,0.0]]) # Covariance is 0 since we know the initial state
positionError, X1, sigma1 = runFilter(1,1, X0, sigma0, pGPS = 1.0)
print(f"\nEstimated State at t=1 after motor command: {X1}")
print(f"Sigma at t=1: {sigma1}")
print(f"\nReal State at t=1: {realX0}")


# ------------------ PLOTTING COMMANDS -------------------- #
plt.figure(2)
plt.scatter(probabilityGPS, positionErrors)
plt.title("Gross Position Error vs. Probability of GPS")
plt.xlabel("Probability of GPS")
plt.ylabel("Absolute Position Error")

plt.figure(3)
timeSteps = np.linspace(0,20,21)
plt.plot(timeSteps, xTotal, label="State Estimation")
plt.plot(timeSteps, rxTotal, label="Real State")
plt.title("State Estimation vs. Real State")
plt.xlabel("Time")
plt.ylabel("Position")
plt.legend(loc='best')

plt.show(block=False)
plt.pause(0.001) # Pause for interval seconds.
input("hit[enter] to end.")
plt.close('all') # all open plots are correctly closed after each run


# ------------------------ PRINT DATA ------------------------ #
np.set_printoptions(precision=2, suppress=True)
print(f"\nWind: {wTotal}")
print(f"rxTotal: {rxTotal}")
print(f"rvTotal: {rvTotal}")
print(f"zTotal: {zTotal}")








