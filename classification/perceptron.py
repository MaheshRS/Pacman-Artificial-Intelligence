# perceptron.py
# -------------
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


# Perceptron implementation
import util
PRINT = True

class PerceptronClassifier:
    """
    Perceptron classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels);
        self.weights = weights;

    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the project description for details.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        (and thus represents a vector a values).
        """

        self.features = trainingData[0].keys() # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

        for iteration in range(self.max_iterations):
            print "Starting iteration ", iteration, "..."
            
            for i in range(len(trainingData)):
                score = None
                high_score_label = None
                
                feature_vector =  trainingData[i];
                
                # For each label compute the score.
                for label in self.legalLabels:
                    local_score = feature_vector * self.weights[label]
                    
                    # If the local score is greater than the high score until now, update the score and the label.
                    if score == None or local_score > score:
                        score = local_score
                        high_score_label = label
                
                # Update the weights appropriately    
                actual_label = trainingLabels[i];
                weight_vector = self.weights[actual_label]
                
                #print len(weight_vector)
                # If the weights are currently zero in 'self.weights', 
                # just update the weight of the label with the features of the test data for the label.
                # else update the weights appropriately.
                if len(weight_vector) is 0:
                    self.weights[actual_label] += feature_vector
                else:
                    if actual_label != high_score_label:
                        self.weights[actual_label] += feature_vector
                        self.weights[high_score_label] -= feature_vector

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


    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """
        featuresWeights = []
        
        weight_vector = self.weights[label];
        sorted_keys = weight_vector.sortedKeys();

        # Add the top 100 feature weights.
        idx = 0
        while idx < 100:
            featuresWeights.append(weight_vector[sorted_keys[idx]])
            idx += 1

        #util.raiseNotDefined()

        #Return the feature weights.
        return featuresWeights
