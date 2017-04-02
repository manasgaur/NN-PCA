The repository contains a report and source code. The report shows the effect of varying
hyperparameters on the accuracy of the neural network. The report quite extensive in its analysis.

The work has been carried out on 200000X 91 dataset of DJI Phantom Drone with 3 classes.
The details are in the attached Question.pdf.

The source code folder contains python programs which should be executed individually.
This is only for education purpose only. If you are using it please make changes to:
1. The file path, which is at the very beginning on the main method.
2. Label adjustment section in the main method. Following is its snippet :
''' Preparing the data'''
    for i in range(len(newdata)):
        ele=newdata[i][0]
        newlst=newdata[i][1:]
        if ele == 1.0:  #second set
            newlst.append(0)
        if ele == -1.0 or ele == 0.0:
            newlst.append(1)
        dataset.append(newlst)

running the program :
python < name of the program> 
