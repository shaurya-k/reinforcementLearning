## get_data_loader(training=True)

Input: an optional boolean argument (default value is True for training dataset)

Return: Dataloader for the training set (if training = True) or the test set (if training = False)

## build_model()

Input: none.

Return: an untrained neural network model

## train_model(model, train_loader, criterion, T)

Input: the model produced by the previous function, the train DataLoader produced by the first function, the criterion, and the number of epochs T for training.

Return: none

## evaluate_model(model, test_loader, criterion, show_loss=True)

Input: the trained model produced by the previous function, the test DataLoader, and the criterion.

It prints the evaluation statistics as described below (displaying the loss metric value if and only if the optional parameter has not been set to False).

Return: none

## predict_label(model, test_images, index)

Input: the trained model and test images,

It prints the top 3 most likely labels for the image at the given index, along with their probabilities.

Return: none
