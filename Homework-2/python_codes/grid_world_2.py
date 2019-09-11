import numpy as np 
from matplotlib import pyplot as plt



class GridWorldClass():

	def __init__(self):

		self.worldSize = 4
		self.discountFactor = 0.9
		
		# self.error = 1e-4
		self.right = [0,1]
		self.left = [0,-1]
		self.up = [-1,0]
		self.down = [1,0]
		self.theta = 1e-4

		self.actionList = [self.left,self.up,self.right,self.down]
		

	
	def _stepFunction(self,action, currentState):
		nextState = [currentState[0]+action[0],currentState[1]+action[1]]


		if (currentState[0]==0 and currentState[1]==0) or (currentState[0]==self.worldSize-1 and currentState[1]==self.worldSize-1):			
			return 0.0,currentState


		if(nextState[0]<0 or nextState[0]>=self.worldSize or nextState[1]<0 or nextState[1]>=self.worldSize):
			nextState=currentState


		rewards =-1.0
		return rewards, nextState

	def policyEvaluation(self,policy):
		
		self.newStateValues =  np.zeros ((self.worldSize,self.worldSize))

		while(True):
			delta  = 0.0


			oldStateValues = self.newStateValues
			
			for currentRow in range(self.worldSize):
				for currentColumn in range(self.worldSize):
					v=self.newStateValues[currentRow,currentColumn]

					currentAction = self.actionList[int(policy[currentRow,currentColumn])]
					newRewards,newState = self._stepFunction(currentAction,[currentRow,currentColumn])
					self.newStateValues[currentRow,currentColumn] = newRewards+self.discountFactor*self.newStateValues[newState[0],newState[1]]
					temp = np.abs(v-self.newStateValues[currentRow,currentColumn])
					delta = max(delta,temp)
					# print(delta)
			
			
			if(delta<self.theta):
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
		# V=self.policyEvaluation(policy)
			
		count=0
		while(True):

			# print("COUNT: ",count)

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
				print("***********FINAL POLICY***********")
				print(policy)
				break

		


	def valueIterations(self):

		self.newStateValues =  np.zeros ((self.worldSize,self.worldSize))

		while(True):
			delta  = 0.0


			oldStateValues = self.newStateValues
			
			for currentRow in range(self.worldSize):
				for currentColumn in range(self.worldSize):
					v=self.newStateValues[currentRow,currentColumn]
					allValues=np.zeros(4)
					for index,currentAction in enumerate(self.actionList):
 

						newRewards,newState = self._stepFunction(currentAction,[currentRow,currentColumn])
						allValues[index] =  newRewards+self.discountFactor*self.newStateValues[newState[0],newState[1]]
						
					self.newStateValues[currentRow,currentColumn] = max(allValues)
					temp = np.abs(v-self.newStateValues[currentRow,currentColumn])
					delta = max(delta,temp)
				
			
			if(delta<self.theta):
				# print("CONVERGE")
				# print(np.round(self.newStateValues,1))
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
		

