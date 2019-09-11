import numpy as np 
from matplotlib import pyplot as plt



class carRental():

	def __init__(self):

		self.maxCars = 20
        self.moxMoveCars
		self.discountFactor = 0.9
		
		self.error = 1e-4
		self.right = [0,1]
		self.left = [0,-1]
		self.up = [-1,0]
		self.down = [1,0]
		self.actionList = [self.left,self.up,self.right,self.down]
		

	
	def _stepFunction(self,action, currentState):
		nextState = [currentState[0]+action[0],currentState[1]+action[1]]


		if (currentState[0]==0 and currentState[1]==0) or (currentState[0]==self.worldSize-1 and currentState[1]==self.worldSize-1):			
			return 10,currentState


		if(nextState[0]<0 or nextState[0]>=self.worldSize or nextState[1]<0 or nextState[1]>=self.worldSize):
			nextState=currentState


		rewards =-1
		return rewards, nextState

	def policyEvaluation(self,policy):
		
		self.newStateValues =  np.zeros ((self.worldSize,self.worldSize))
		
		while(True):

			oldStateValues = self.newStateValues
			for currentRow in range(self.worldSize):
				for currentColumn in range(self.worldSize):

					currentAction = self.actionList[int(policy[currentRow,currentColumn])]
					

					newRewards,newState = self._stepFunction(currentAction,[currentRow,currentColumn])

					self.newStateValues[currentRow,currentColumn] = newRewards+self.discountFactor*self.newStateValues[newState[0],newState[1]]
			delta = abs(oldStateValues-self.newStateValues).max()
				
			if(delta<self.error):
				# print("CONVERGE")
				# print(np.round(self.newStateValues,1))
				return self.newStateValues
				
				
	
	def oneStepLookAheaFunction(self,currentRow,currentColumn,V):
		
		allValues=np.zeros(4)
		for currentAction in range(4):
			newRewards,newState = self._stepFunction(self.actionList[currentAction],[currentRow,currentColumn])
					
			allValues[currentAction]= newRewards+self.discountFactor*V[newState[0],newState[1]]
		return np.argmax(allValues)


	
	def policyImprovement(self):

		policy = np.zeros((4,4))
		actionIndex = [0,1,2,3]

		for currentRow in range(self.worldSize):
			for currentColumn in range(self.worldSize):
				policy[currentRow,currentColumn]=np.random.choice(actionIndex)

			

		print("INTIAL POLICY:\n",policy)
		print("*************************")
		
		count=0
		while(True):

			print("COUNT: ",count)

			count+=1
			V=self.policyEvaluation(policy)
			policyStable =True
			tempCount=0
			for currentRow in range(self.worldSize):
				for currentColumn in range(self.worldSize):

					chosenAction =policy[currentRow,currentColumn]
					actionTaken = self.oneStepLookAheaFunction(currentRow,currentColumn,V)
					
					policy[currentRow,currentColumn] = actionTaken 
					if(chosenAction!=actionTaken):
						tempCount+=1
						policyStable=False
					
			if(policyStable):
				print(policy)
				return policy ,V
		


	def valueIterations(self):

		self.newStateValues =  np.zeros ((self.worldSize,self.worldSize))
			
		while(True):

			oldStateValues = self.newStateValues
			for currentRow in range(self.worldSize):
				for currentColumn in range(self.worldSize):
					allValues=np.zeros(4)
					for index,currentAction in enumerate(self.actionList):

						newRewards,newState = self._stepFunction(currentAction,[currentRow,currentColumn])
						allValues[index] = newRewards+self.discountFactor*self.newStateValues[newState[0],newState[1]]

					self.newStateValues[currentRow,currentColumn] = np.max(allValues)
			delta = abs(oldStateValues-self.newStateValues).max()
				
			if(delta<self.error):
				print("CONVERGE")
				print(np.round(self.newStateValues,1))
				break
				
		policy = np.zeros((4,4))
			

		print("INTIAL POLICY:\n",policy)
		print("*************************")

		for currentRow in range(self.worldSize):
			for currentColumn in range(self.worldSize):
				allValues=np.zeros(4)
				for index,currentAction in enumerate(self.actionList):

					newRewards,newState = self._stepFunction(currentAction,[currentRow,currentColumn])
					allValues[index] = newRewards+self.discountFactor*self.newStateValues[newState[0],newState[1]]

			policy[currentRow,currentColumn] = np.argmax(allValues)
		
		
		print(policy)
		
