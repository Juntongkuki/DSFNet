
API Reference
===========================

+-----------------------------+-----------------------------------------+
| ``tnlearn.VecSymRegressor`` | Symbolic Regression with Regressor Class|
+-----------------------------+-----------------------------------------+
| ``tnlearn.MLPClassifier``   | MLPClassifier Class Implementation      |
+-----------------------------+-----------------------------------------+
| ``tnlearn.MLPRegressor``    | MLPRegressor Class Implementation       |
+-----------------------------+-----------------------------------------+




.. role:: classtext
   :class: classtext


tnlearn.VecSymRegressor
----------------
- Description:
    ``tnlearn.VecSymRegressor`` implements the symbolic regression algorithm through the Regressor class, enabling the evolution of mathematical expressions to fit given data. The class provides methods for generating random expressions, evaluating their fitness, and evolving expressions through mutation and crossover. It aims to find the best-fitting mathematical model for a given dataset.

    .. container:: custom-background

       ``tnlearn.VecSymRegressor`` : :classtext:`class` **tnlearn.VecSymRegressor** :classtext:`(random_state=100, pop_size=5000, max_generations=20, tournament_size=10, x_pct=0.7, xover_pct=0.3, save=True, operations=None)`


- **How to initialize your** ``VecSymRegressor`` **:**
    ``random_state``：Seed for random number generation **(default: 100)**

    ``pop_size``：Population size for genetic algorithm **(default: 5000)**

    ``max_generations``：Maximum generations for genetic algorithm **(default: 20)**

    ``tournament_size``：Size of tournament selection **(default: 10)**

    ``x_pct``：Probability of selecting a variable node during random program generation **(default: 0.7)**

    ``xover_pct``：Crossover probability during offspring generation **(default: 0.3)**

    ``save``：Flag for saving **(default: False)**

        .. container:: custom-background-2

            **False:** Don't need to save the progress

            **True:** If save option is enabled, open a file called **"log.txt"** to log the progress

    ``operations``：Set of operations to be used in program generation **(default: None)**

        .. container:: custom-background-2

            **None:** Use the defult mathematical operations for the algorithm, like the following code:

            .. code-block:: python
                :linenos:

                (
                {"func": operator.add, "arg_count": 2, "format_str": "({} + {})"},
                {"func": operator.sub, "arg_count": 2, "format_str": "({} - {})"},
                {"func": operator.mul, "arg_count": 2, "format_str": "({} * {})"},
                {"func": operator.neg, "arg_count": 1, "format_str": "-({})"},
                )
            You can also define ``operations`` in a similar way.



- **Here is an example of using** ``VecSymRegressor`` **quickly:**

    .. code-block::

        from tnlearn import Regressor

        >> neuron = Regressor()
        >> neuron.fit(X_train, y_label)
        >> print('*' * 20)
        >> print(neuron.neuron)

        ********************
        6@x**2 + 1.46@x - 0.1316

    .. note::
        Format ``X_train`` and ``y_label`` as **numpy arrays**.

        In the above example, the shapes of ``X_train`` and ``y_label`` are **(600, 10)** and **(600, 1)** respectively.



tnlearn.MLPClassifier
----------------------
- Description:
    ``tnlearn.MLPClassifier`` implements the class MLPClassifier, which extends the functionality of the base class to build and train a Multi-layer Perceptron (MLP) model. MLPClassifier is designed to allow easy customization of the neural network structure, activation functions, and loss function used during training. It incorporates device selection to leverage available GPU resources, ensuring efficient computation. The class covers essential methods for model training, evaluation, and prediction, making it a flexible tool for supervised learning tasks in PyTorch.

    .. container:: custom-background

       ``tnlearn.MLPClassifier`` : :classtext:`class` **tnlearn.MLPClassifier** :classtext:`(neurons='x', layers_list=None, activation_funcs=None, loss_function=None, optimizer_name='adam', random_state=1, max_iter=300, batch_size=128, valid_size=0.2, lr=0.01, visual=False, save=False, visual_interval=100, gpu=None, interval=None, scheduler=None, l1_reg=None, l2_reg=None)`


- **How to initialize your** ``tnlearn.MLPClassifier`` **:**
    ``neurons``: Neuronal expression **(default: 'x')**

        .. container:: custom-background-2

            Users can pass the results of ``tnlearn.VecSymRegressor`` to ``neurons``

    ``layers_list``: List of neuron counts for each hidden layer **(default: [50, 30, 10])**

    ``activation_funcs``: Activation functions **(default: None)**

        .. container:: custom-background-2

            Users can choose different activation functions: ``'relu'``, ``'leakyrelu'``, ``'sigmoid'``, ``'tanh'``,  ``'softmax'``

    ``loss_function``: Loss function for the training process **(default: None)**

        .. container:: custom-background-2

            Users can choose different loss functions: ``'mse'``, ``'l1'``, ``'crossentropy'``, ``'bce'``

    ``optimizer_name``: Name of the optimizer algorithm **(default: 'adam')**

        .. container:: custom-background-2

            Users can choose different optimizers: ``'adam'``, ``'sgd'``, ``'rmsprop'``, ``'adamw'``

    ``random_state``: Seed for random number generators for reproducibility **(default: 1)**

    ``max_iter``: Maximum number of training iterations **(default: 300)**

    ``batch_size``: Number of samples per batch during training **(default: 128)**

    ``valid_size``: Fraction of training data used for validation **(default: 0.2)**

    ``lr``: Learning rate for the optimizer **(default: 0.01)**

    ``visual``: Boolean indicating if training visualization is to be shown **(default: False)**

    ``save``: Indicates if the training figure should be saved **(default: False)**

    ``visual_interval``: Interval at which training visualization is updated **(default: 100)**

    ``gpu``: Specifies GPU configuration for training **(default: None)**

        .. container:: custom-background-2

            **None**: Not use GPU

            **An Integer (e.g. 1)**: How many GPUs you want to use

    ``interval``: Interval of screen output during training **(default: None)**

    ``scheduler``: Learning rate scheduler **(default: None)**

        .. container:: custom-background-2

            **None**:  Not use any learning rate adjustment strategy

            **{'step_size': 30, 'gamma': 0.2}**:  Use ``lr_sceduler.StepLR()`` with **"step_size = 30"** and **"gamma = 0.2"**

    ``l1_reg``: L1 regularization term **(default: None)**

        .. container:: custom-background-2

            **None**:  Not use L1 regularization

            **True**:  Use L1 regularization

    ``l2_reg``: L2 regularization term **(default: None)**

        .. container:: custom-background-2

            **None**:  Not use L2 regularization

            **True**:  Use L2 regularization


- **Here is an example of using** ``MLPClassifier`` **quickly:**

    .. code-block:: python
        :linenos:

        from tnlearn import MLPClassifier
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from tnlearn import VecSymRegressor

        X, y = make_classification(n_samples=200, random_state=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

        neuron = VecSymRegressor()

        neuron.fit(X_train, y_train)
        print('*' * 20)
        print(neuron.neuron)

        layers_list = [10, 10, 10]
        clf = MLPClassifier()

        clf.fit(X_train, y_train)
        print(f"Score: {clf.score(X_test, y_test)}")

        clf.save(path='my_model_dir', filename='mlp_classifier.pth')
        clf.load(path='my_model_dir', filename='mlp_classifier.pth', input_dim=X_train.shape[1], output_dim=1)
        clf.fit(X_train, y_train)


tnlearn.MLPRegressor
--------------------
- Description:
    ``tnlearn.MLPRegressor`` embodies the MLPRegressor class, designed as an extension of BaseModel within a custom machine learning framework. It facilitates the creation and training of a multilayer perceptron (MLP) for regression tasks. Features of the class include the flexibility to define custom neural network architectures through parameters such as neuronal expression and layer structure, the utilization of various activation functions, and the incorporation of modern optimization algorithms and regularization techniques. Moreover, the class is equipped with optional GPU support for enhanced computational efficiency, as well as functionalitiesfor training visualization, validation, and model performance evaluation. This representation of an MLP is tailored to adapt to an array of regression problems while ensuring ease of use and extensibility.

    .. container:: custom-background

       ``tnlearn.MLPRegressor`` : :classtext:`class` **tnlearn.MLPRegressor** :classtext:`(neurons='x', layers_list=None, activation_funcs=None, loss_function=None, optimizer_name='adam', random_state=1, max_iter=300, batch_size=128, valid_size=0.2, lr=0.01, visual=False, save=False, visual_interval=100, gpu=None, interval=None, scheduler=None, l1_reg=None, l2_reg=None)`


- **How to initialize your** ``tnlearn.MLPRegressor`` **:**
    ``neurons``: Neuronal expression **(default: 'x')**

        .. container:: custom-background-2

            Users can pass the results of ``tnlearn.VecSymRegressor`` to ``neurons``

    ``layers_list``: List of neuron counts for each hidden layer **(default: [50, 30, 10])**

    ``activation_funcs``: Activation functions **(default: None)**

        .. container:: custom-background-2

            Users can choose different activation functions: ``'relu'``, ``'leakyrelu'``, ``'sigmoid'``, ``'tanh'``,  ``'softmax'``

    ``loss_function``: Loss function for the training process **(default: None)**

        .. container:: custom-background-2

            Users can choose different loss functions: ``'mse'``, ``'l1'``, ``'crossentropy'``, ``'bce'``

    ``optimizer_name``: Name of the optimizer algorithm **(default: 'adam')**

        .. container:: custom-background-2

            Users can choose different optimizers: ``'adam'``, ``'sgd'``, ``'rmsprop'``, ``'adamw'``

    ``random_state``: Seed for random number generators for reproducibility **(default: 1)**

    ``max_iter``: Maximum number of training iterations **(default: 300)**

    ``batch_size``: Number of samples per batch during training **(default: 128)**

    ``valid_size``: Fraction of training data used for validation **(default: 0.2)**

    ``lr``: Learning rate for the optimizer **(default: 0.01)**

    ``visual``: Boolean indicating if training visualization is to be shown **(default: False)**

    ``save``: Indicates if the training figure should be saved **(default: False)**

    ``visual_interval``: Interval at which training visualization is updated **(default: 100)**

    ``gpu``: Specifies GPU configuration for training **(default: None)**

        .. container:: custom-background-2

            **None**: Not use GPU

            **An Integer (e.g. 1)**: How many GPUs you want to use

    ``interval``: Interval of screen output during training **(default: None)**

    ``scheduler``: Learning rate scheduler **(default: None)**

        .. container:: custom-background-2

            **None**:  Not use any learning rate adjustment strategy

            **{'step_size': 30, 'gamma': 0.2}**:  Use ``lr_sceduler.StepLR()`` with **"step_size = 30"** and **"gamma = 0.2"**

    ``l1_reg``: L1 regularization term **(default: None)**

        .. container:: custom-background-2

            **None**:  Not use L1 regularization

            **True**:  Use L1 regularization

    ``l2_reg``: L2 regularization term **(default: None)**

        .. container:: custom-background-2

            **None**:  Not use L2 regularization

            **True**:  Use L2 regularization


- **Here is an example of using** ``MLPRegressor`` **quickly:**

    .. code-block:: python
        :linenos:

        from tnlearn import MLPRegressor
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from tnlearn import VecSymRegressor

        X, y = make_classification(n_samples=200, random_state=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

        neuron = VecSymRegressor()

        neuron.fit(X_train, y_train)
        print('*' * 20)
        print(neuron.neuron)

        layers_list = [10, 10, 10]
        clf = MLPRegressor()

        clf.fit(X_train, y_train)
        print(f"Score: {clf.score(X_test, y_test)}")

        clf.save(path='my_model_dir', filename='mlp_regressor.pth')
        clf.load(path='my_model_dir', filename='mlp_regressor.pth', input_dim=X_train.shape[1], output_dim=1)
        clf.fit(X_train, y_train)