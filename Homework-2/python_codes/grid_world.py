import numpy as np 
from matplotlib import pyplot as plt



class GridWorldClass():

	def __init__(self):

		self.worldSize = 5
		self.discountFactor = 0.9
		self.totalEpochs =1000
		self.error = 1e-9
		self.Aposition = [0,1]
		self.APrimePosition = [4,1]
		self.BPosition = [0,3]
		self.BPrimePosition = [2,3]
		self.right = [0,1]
		self.left = [0,-1]
		self.up = [-1,0]
		self.down = [1,0]
		self.actionList = [self.right,self.left,self.up,self.down]

	def _stepFunction(self,action, currentState):


		if(currentState ==self.Aposition):
			nextState = self.APrimePosition
			rewards = +10
		 
		elif(currentState ==self.BPosition):
			nextState = self.BPrimePosition
			rewards = +5
		
		else:
			nextState = [currentState[0]+action[0],currentState[1]+action[1]]
			rewards = 0

			if(nextState[0]<0 or nextState[0]>=self.worldSize or nextState[1]<0 or nextState[1]>=self.worldSize):
				rewards =-1
				nextState=currentState
				

		return rewards, nextState



	def simulateLinear(self):
		grid = np.zeros ((self.worldSize,self.worldSize))
		print(grid)
		while(True):
			newGrid = np.zeros ((self.worldSize,self.worldSize))
			# print("EPOCH: "+str(currentEpoch))
			for currentRow in range(self.worldSize):
				for currentColumn in range(self.worldSize):
					for currentAction in self.actionList:
						newRewards,newState = self._stepFunction(currentAction,[currentRow,currentColumn])
						newGrid[currentRow,currentColumn]+= 0.25 * (newRewards+self.discountFactor*grid[newState[0],newState[1]])
			
			if(np.sum(np.abs(newGrid-grid))<self.error):
				print(np.round(grid,1))
				break
				
			else:
				grid = newGrid
		 
		
	def simulateOptimal(self):
		grid = np.zeros ((self.worldSize,self.worldSize))
		print(grid)
		while(True):
			newGrid = np.zeros ((self.worldSize,self.worldSize))
			# print("EPOCH: "+str(currentEpoch))
			for currentRow in range(self.worldSize):
				for currentColumn in range(self.worldSize):
					temp=[]
					for currentAction in self.actionList:
						newRewards,newState = self._stepFunction(currentAction,[currentRow,currentColumn])
						temp.append( newRewards+self.discountFactor*grid[newState[0],newState[1]])
					newGrid[currentRow,currentColumn]=np.max(temp)
			if(np.sum(np.abs(newGrid-grid))<self.error):
				print(np.round(grid,1))
				break
			else:
				grid = newGrid
		 
