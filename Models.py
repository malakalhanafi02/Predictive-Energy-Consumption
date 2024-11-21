from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM, GRU
from tensorflow.keras.models import Sequential
import re

def build_fully_connected_network(layer_list, input_shape=None):
    """
    Build a fully connected neural network based on a provided layer list.

    Parameters:
    layer_list (list): A list specifying the layers to add. Examples:
        [10, 'Dropout(0.1)', 10, 'Dropout(0.1)', 1]
        The last Dense layer in a regression problem will use 'linear' activation.

    input_shape (tuple or None): Shape of the input data for the first layer.
        Required for the first Dense layer if it's the first layer of the model.

    Returns:
    model: Compiled Keras Sequential model.
    """
    model = Sequential()
    
    for i, layer in enumerate(layer_list):
        if isinstance(layer, int):
            # Check if it's the last layer to set activation for regression
            if i == len(layer_list) - 1:
                activation = 'linear'  # Final layer in regression
            else:
                activation = 'relu'
            
            if i == 0 and input_shape is not None:
                model.add(Dense(layer, activation=activation, input_shape=input_shape))
            else:
                model.add(Dense(layer, activation=activation))
        elif isinstance(layer, str) and "Dropout" in layer:
            # Parse the dropout rate from the string (e.g., 'Dropout(0.1)')
            match = re.match(r"Dropout\((.*)\)", layer)
            if match:
                rate = float(match.group(1))
                model.add(Dropout(rate))
            else:
                raise ValueError(f"Invalid layer specification: {layer}")
        else:
            raise ValueError(f"Unsupported layer type: {layer}")
    
    return model

def build_1d_cnn(layer_list, input_shape=None):
    """
    Build a 1D CNN based on a provided layer list.

    Parameters:
    layer_list (list): A list specifying the layers to add. Examples:
        [
            'Conv1D(32, 3)', 'MaxPooling1D(2)', 
            'Conv1D(64, 3)', 'MaxPooling1D(2)',
            'Flatten', 128, 'Dropout(0.5)', 1
        ]
        - Conv1D(32, 3): Convolutional layer with 32 filters and kernel size 3.
        - MaxPooling1D(2): Max pooling with pool size 2.
        - Flatten: Flattens the 1D output into 1D for Dense layers.
        - 128: Dense layer with 128 units.
        - Dropout(0.5): Dropout with 50% rate.
        - 1: Dense output layer (e.g., regression with linear activation).

    input_shape (tuple): Shape of the input data (required for the first layer).

    Returns:
    model: Compiled Keras Sequential model.
    """
    model = Sequential()
    
    for i, layer in enumerate(layer_list):
        if isinstance(layer, str):
            # Parse layer specification
            if "Conv1D" in layer:
                match = re.match(r"Conv1D\((\d+),\s*(\d+)\)", layer)
                if match:
                    filters = int(match.group(1))
                    kernel_size = int(match.group(2))
                    if i == 0 and input_shape is not None:
                        model.add(Conv1D(filters, kernel_size, activation='relu', input_shape=input_shape))
                    else:
                        model.add(Conv1D(filters, kernel_size, activation='relu'))
                else:
                    raise ValueError(f"Invalid Conv1D layer specification: {layer}")
            elif "MaxPooling1D" in layer:
                match = re.match(r"MaxPooling1D\((\d+)\)", layer)
                if match:
                    pool_size = int(match.group(1))
                    model.add(MaxPooling1D(pool_size))
                else:
                    raise ValueError(f"Invalid MaxPooling1D layer specification: {layer}")
            elif "Flatten" == layer:
                model.add(Flatten())
            elif "Dropout" in layer:
                match = re.match(r"Dropout\((.*)\)", layer)
                if match:
                    rate = float(match.group(1))
                    model.add(Dropout(rate))
                else:
                    raise ValueError(f"Invalid Dropout layer specification: {layer}")
            else:
                raise ValueError(f"Unsupported layer type: {layer}")
        elif isinstance(layer, int):
            # Add a Dense layer
            if i == len(layer_list) - 1:
                activation = 'linear'  # Final layer in regression
            else:
                activation = 'relu'
            if i == 0 and input_shape is not None:
                model.add(Dense(layer, activation=activation, input_shape=input_shape))
            else:
                model.add(Dense(layer, activation=activation))
        else:
            raise ValueError(f"Unsupported layer type: {layer}")
    
    return model

def build_rnn(layer_list, input_shape=None):
    """
    Build an RNN based on a provided layer list.

    Parameters:
    layer_list (list): A list specifying the layers to add. Examples:
        [
            'LSTM(50)', 'Dropout(0.2)', 
            'GRU(30)', 'Dropout(0.3)',
            128, 'Dropout(0.5)', 1
        ]
        - LSTM(50): LSTM layer with 50 units.
        - GRU(30): GRU layer with 30 units.
        - Dropout(0.2): Dropout with 20% rate.
        - 128: Dense layer with 128 units.
        - 1: Dense output layer (e.g., regression with linear activation).

    input_shape (tuple): Shape of the input data (required for the first layer).

    Returns:
    model: Compiled Keras Sequential model.
    """
    model = Sequential()
    
    for i, layer in enumerate(layer_list):
        if isinstance(layer, str):
            # Parse layer specification
            if "LSTM" in layer:
                match = re.match(r"LSTM\((\d+)\)", layer)
                if match:
                    units = int(match.group(1))
                    if i == 0 and input_shape is not None:
                        model.add(LSTM(units, activation='tanh', recurrent_activation='sigmoid', input_shape=input_shape, return_sequences=True))
                    else:
                        model.add(LSTM(units, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
            elif "GRU" in layer:
                match = re.match(r"GRU\((\d+)\)", layer)
                if match:
                    units = int(match.group(1))
                    if i == 0 and input_shape is not None:
                        model.add(GRU(units, activation='tanh', recurrent_activation='sigmoid', input_shape=input_shape, return_sequences=True))
                    else:
                        model.add(GRU(units, activation='tanh', recurrent_activation='sigmoid', return_sequences=True))
            elif "Dropout" in layer:
                match = re.match(r"Dropout\((.*)\)", layer)
                if match:
                    rate = float(match.group(1))
                    model.add(Dropout(rate))
                else:
                    raise ValueError(f"Invalid Dropout layer specification: {layer}")
            else:
                raise ValueError(f"Unsupported layer type: {layer}")
        elif isinstance(layer, int):
            # Add a Dense layer
            if i == len(layer_list) - 1:
                activation = 'linear'  # Final layer in regression
            else:
                activation = 'relu'
            if i == 0 and input_shape is not None:
                model.add(Dense(layer, activation=activation, input_shape=input_shape))
            else:
                model.add(Dense(layer, activation=activation))
        else:
            raise ValueError(f"Unsupported layer type: {layer}")
    
    # Ensure the final recurrent layer does not return sequences if followed by Dense
    for j in range(len(model.layers) - 1, -1, -1):
        if isinstance(model.layers[j], (LSTM, GRU)) and j < len(model.layers) - 1:
            model.layers[j].return_sequences = False
            break
    
    return model

'''
#Example of a Neural Network
# Define the desired architecture
layer_list = [64, 'Dropout(0.2)', 64, 'Dropout(0.2)', 1]  # Final output neuron for regression

# Specify the input shape
input_shape = (10,)  # Assuming 10 input features

# Build the model
model = build_fully_connected_network(layer_list, input_shape=input_shape)

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Display the model summary
model.summary()
'''

'''
#Example of a CNN then Neural Network
# Define the architecture
layer_list = [
    'Conv1D(32, 3)', 'MaxPooling1D(2)',
    'Conv1D(64, 3)', 'MaxPooling1D(2)',
    'Flatten', 128, 'Dropout(0.5)', 1
]

# Input shape for 1D data, e.g., (100, 1) for 100 time steps with 1 feature
input_shape = (100, 1)

# Build the model
model = build_1d_cnn(layer_list, input_shape=input_shape)

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Display the model summary
model.summary()
'''

'''
# Example for LSTM and GRU
# Define the architecture
layer_list = [
    'LSTM(64)', 'Dropout(0.2)', 
    'LSTM(64)', 'Dropout(0.2)', 
    'GRU(32)', 'Dropout(0.3)', 
    'GRU(32)', 'Dropout(0.3)', 
    128, 'Dropout(0.1)', 1
]

# Input shape for sequential data, e.g.,
input_shape = (100,1)

# Build the model
model = build_rnn(layer_list, input_shape)

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Display the model summary
model.summary()
'''