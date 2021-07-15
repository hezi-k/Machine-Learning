import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def fPC (y, yhat):
    return np.mean(y == yhat) 

def measureAccuracyOfPredictors (predictors, X, y):

    total = np.zeros(y.shape)

    for predictor in predictors:

        r1, c1, r2, c2 = predictor

        # r1 = predictor[0]
        # c1 = predictor[1]
        # r2 = predictor[2]
        # c2 = predictor[3]
        
        comparison = X[:,r1,c1] > X[:,r2,c2]

        #converted = comparison.astype(int)

        converted = np.zeros(X.shape[0])
        converted[comparison] = 1

        total += converted

    averagedPredictors = total/len(predictors)

    averagedPredictors[averagedPredictors > 0.5] = 1
    averagedPredictors[averagedPredictors <= 0.5] = 0

    return fPC(y, averagedPredictors)


     

def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels): 

    predictors = []

    for m in range(0,5):

        bestSoFar = 0
        goodPixels = (0,0,0,0)

        for r1 in range(0,24):
            for c1 in range(0,24):
                for r2 in range(0,24):
                    for c2 in range(0,24):

                        if(r1 == r2 and c1 == c2):
                            continue

                        
                        # ensemble = predictors
                        # ensemble.append((r1,c1,r2,c2))

                        #ensemble = predictors + list(((r1, c1, r2, c2),))

                        currentPredictionAccuracy = measureAccuracyOfPredictors(predictors + [(r1, c1, r2, c2)], trainingFaces, trainingLabels)

                        if(currentPredictionAccuracy > bestSoFar):
                            bestSoFar = currentPredictionAccuracy
                            goodPixels = (r1,c1,r2,c2)

        predictors.append(goodPixels)

    #r1,c1,r2,c2 = goodPixels

    #print(predictors)

    return predictors

def analyseAccuracy(trainingFaces, trainingLabels, testingFaces, testingLabels):

    n = [400, 800, 1200, 1600, 2000]
    features = []

    for x in n:
        print("Analysing Training Accuracy for n =", x, "\n")
        features = stepwiseRegression(trainingFaces[:x],trainingLabels [:x],testingFaces,testingLabels)
        trainingAccuracy = measureAccuracyOfPredictors(features, trainingFaces[:x],trainingLabels[:x])
        print("Accuracy on Training Set: ", trainingAccuracy)
        print("Analysing Testing Accuracy for n =", x, "\n")
        testingAccuracy = measureAccuracyOfPredictors(features, testingFaces, testingLabels)
        print("Accuracy on Testing Set: ", testingAccuracy)


    
    visualiseFeatures(features, testingFaces, trainingFaces)


    
def visualiseFeatures(predictors, testingFaces, trainingFaces):
    show = True
    if show:
        # Show an arbitrary test image in grayscale
        im = testingFaces[0,:,:]
        im2 = trainingFaces[0,:,:]
        fig,ax = plt.subplots(1)
        ax.imshow(im, cmap='gray')
        # Show r1,c1

        for predictor in predictors:

            r1, c1, r2, c2 = predictor

            rect = patches.Rectangle((c1 - 0.5, r1 - 0.5), 1, 1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            # Show r2,c2
            rect = patches.Rectangle((c2 - 0.5, r2 - 0.5), 1, 1, linewidth=2, edgecolor='b', facecolor='none')
            ax.add_patch(rect)
        # Display the merged result
        plt.show()

def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    analyseAccuracy(trainingFaces,trainingLabels, testingFaces, testingLabels)
    #print(stepwiseRegression(trainingFaces[:400],trainingLabels [:400],testingFaces,testingLabels))