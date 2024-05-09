# deep-learning-challenge

This Repo analyses data from a csv file containing data over the success of funding attempts from organisations. Through the creation of machine learning and neural networks, this repo attempts to predict whether new attempts will be successful. 

------------------------------------------
Analysis:
Aim- To predict, with 75% accuracy, whether new funding campaigns will be successful
Process- Reduce irrelevant data, bin columns with excessive unique values, convert categorical data into binary/numerical values. Train dataset against [IS_SUCCESSFUL] column before identifying optimal neural network. Finally, increase epoch value of neural network to discover accuracy value.

- Columns removed with beneficial results: [EIN, NAME, SPECIAL_CONSIDERATIONS]. [STATUS] was removed with very small difference
- Target: [SUCCESSFUL]
- Feature variables: [APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, ASK_AMT]

- The initial deep learning model was as seen below. This was based on the KerasTuner optimization parameters that was given:
'activation': 'sigmoid',
 'first_units': 26,
 'num_layers': 3,
 'units_0': 16,
 'units_1': 16,
 'units_2': 16

Trial 1:
Unique parameters-
Columns removed: [EIN, NAME]
Bins added: [CLASSIFICATION] <700, [APPLICATION_TYPE] < 500
KerasTuner: Activation choices = [relu, tanh, sigmoid]. Layers = max 5, neurons = max 30, epochs = 20
Deep learning model based on optimal KerasTuner: epochs up to 100

Results-
Through these parameters, the optimal neural network gave a highest accuracy of 74.27 at epoch 96

Trial 2:
Unique parameters-
Columns removed: [EIN, NAME, AFFILIATION]
Bins added: [CLASSIFICATION] <150, [APPLICATION_TYPE] < 100
KerasTuner: Activation choices = [relu, tanh, sigmoid]. Layers = max 5, neurons = max 40, epochs = 20
Deep learning model based on optimal KerasTuner: epochs up to 100

Results-
This trial performed particularly poorly, with highest KerasTuner results of 66%. Due to the poor results, the deep learning model was not implemented. Through this, it is clear that the [APPLICATION_TYPE] is fundamental to the accuracy.


Trial 3:
Unique parameters-
Columns removed: [EIN, NAME, SPECIAL_CONSIDERATIONS, STATUS]
Bins added: [CLASSIFICATION] <150, [APPLICATION_TYPE] < 100
KerasTuner: Activation choices = ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu', 'selu', 'swish', 'prelu']. Layers = max 10, neurons = max 40, epochs = 20
Deep learning model based on optimal KerasTuner: epochs up to 120

Results-
The optimal model running on 120 epochs returned an accuracy of 74.5%. This result is mixed regarding hitting the goal of 75%. At a base level, it is below the 75 mark, however, rounding to the nearest whole number would constitute a value of 75%. This model used RELU, so no added activations were relevant to this model. The rise should therefore be attributed to the elimination of the special considerations and status columns, or the heightened neuron/layer parameters

Overall results:
The goal of 75% was arguable reached on trial 3, however, a slightly higher value would be needed to firmly reach the goal

Further Notes:
HDF5 file based on output from trial_3