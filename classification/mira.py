# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        "*** YOUR CODE HERE ***"
        optimal_weights = util.Counter();
        optimal_c = -1
        
        for c_value in Cgrid:
            local_weights = self.weights.copy()
            for iteration in range(self.max_iterations):
                print "Starting iteration ", iteration, "...", " Cgrid Value: ", c_value
                
                for i in range(len(trainingData)):
                    score = None
                    high_score_label = None
            
                    feature_vector =  trainingData[i].copy();
            
                    # For each label compute the score.
                    for label in self.legalLabels:
                        local_score = feature_vector * local_weights[label]
            
                        # If the local score is greater than the high score until now, update the score and the label.
                        if score == None or local_score > score:
                            score = local_score
                            high_score_label = label
            
                    # Update the weights appropriately    
                    actual_label = trainingLabels[i];
                    weight_vector = local_weights[actual_label]
            
                    #print len(weight_vector)
                    # If the weights are currently zero in 'self.weights', 
                    # just update the weight of the label with the features of the test data for the label.
                    # else update the weights appropriately.
                    
                    alpha = (((local_weights[high_score_label] - local_weights[actual_label]) * feature_vector) + 1.)/(2 * (feature_vector * feature_vector))
                    feature_vector.divideAll(1.0/alpha)
                    
                    if actual_label != high_score_label:
                        local_weights[actual_label] += feature_vector
                        local_weights[high_score_label] -= feature_vector;
            
            validation_score_sum = 0
            validation_guesses = self.classify(validationData)
            for j, guess in enumerate(validation_guesses):
                validation_score_sum += (validationLabels[j] == guess and 1.0 or 0.0)
            
            if validation_score_sum > optimal_c:
                optimal_c = validation_score_sum
                optimal_weights[optimal_c] = local_weights
                    
        self.weights = optimal_weights[optimal_c]
        #util.raiseNotDefined()

    def classify(self, data ):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


