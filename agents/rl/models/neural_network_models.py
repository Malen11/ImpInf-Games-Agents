# -*- coding: utf-8 -*-

import tensorflow as tf

class SimpleNeuralNetworkModel(tf.keras.Model):
      
    def __init__(self, 
                 num_input, 
                 hidden_units, 
                 num_output, 
                 activation_func='tanh', 
                 kernel_initializer='RandomNormal',
                 output_activation_func='tanh', 
                 output_kernel_initializer='RandomNormal',
                 **kwargs):
        '''
        
        Инициализация модели сети

        Parameters
        ----------
        num_input : int
            Количество параметров входного слоя.
        hidden_units : [init]
            Массив размерности скрытых слоёв.
        num_output: [init]
            Массив размерности выходного слоя.
        activation_func : str, optional
            Функция активации сети. The default is 'tanh'.
        kernel_initializer : str, optional
            DESCRIPTION. The default is 'RandomNormal'.
        output_activation_func : str, optional
            DESCRIPTION. The default is 'tanh'.
        output_kernel_initializer : str, optional
            DESCRIPTION. The default is 'RandomNormal'.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        super(SimpleNeuralNetworkModel, self).__init__(**kwargs)
        
        self.num_input = num_input
        self.hidden_units = hidden_units
        self.num_output = num_output
        self.activation_func = activation_func
        self.kernel_initializer = kernel_initializer
        self.output_activation_func = output_activation_func
        self.output_kernel_initializer = output_kernel_initializer
        
        #создание входного слоя сети
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(num_input,))
        
        #создание скрытых слоёв сети
        self.hidden_layers = []
    
        for i in hidden_units:
            self.hidden_layers.append(SimpleNeuralNetworkLayerBlock(
                i, 
                activation_func=activation_func, 
                kernel_initializer=kernel_initializer))
            
        #создание выходного слоя сети
        self.output_layer = tf.keras.layers.Dense(
            num_output, 
            activation=output_activation_func, 
            kernel_initializer=output_kernel_initializer)
        
        
    @tf.function
    def call(self, inputs, training=None):
        '''
        Расчёт значений модели

        Parameters
        ----------
        inputs : TYPE
            Входные данные сети(состояние).
        training : TYPE
            режим тренировки.

        Returns
        -------
        output : TYPE
            DESCRIPTION.

        '''
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
    
    def get_config(self):
        
        config = super(SimpleNeuralNetworkModel, self).get_config()
        config.update({'num_input': self.num_input,
                       'hidden_units': self.hidden_units,
                       'num_output': self.num_output,
                       'activation_func': self.activation_func,
                       'kernel_initializer': self.kernel_initializer,
                       'output_activation_func': self.output_activation_func,
                       'output_kernel_initializer': self.output_kernel_initializer})
        return config
    
class LSTMNeuralNetworkModel(tf.keras.Model):
    
    def __init__(self, 
                 num_input,
                 lstm_units, 
                 hidden_units, 
                 num_output,
                 timesteps,
                 activation_func='tanh', 
                 kernel_initializer='RandomNormal',
                 output_activation_func='tanh', 
                 output_kernel_initializer='RandomNormal',
                 **kwargs):
        super(LSTMNeuralNetworkModel, self).__init__(**kwargs)
        
        self.num_input = num_input
        self.lstm_units = lstm_units
        self.hidden_units = hidden_units
        self.num_output = num_output
        self.timesteps = timesteps
        self.activation_func = activation_func
        self.kernel_initializer = kernel_initializer
        self.output_activation_func = output_activation_func
        self.output_kernel_initializer = output_kernel_initializer
        
        #создание входного слоя сети
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(timesteps, num_input))
        
        #lstm слой
        self.lstm_layers  = []
        
        for i in lstm_units:
            self.lstm_layers.append(tf.keras.layers.LSTM(i))
            
        #создание скрытых слоёв сети
        self.hidden_layers = []
    
        for i in hidden_units:
            self.hidden_layers.append(SimpleNeuralNetworkLayerBlock(
                i, 
                activation_func=activation_func, 
                kernel_initializer=kernel_initializer))
            
        #создание выходного слоя сети
        self.output_layer = tf.keras.layers.Dense(
            num_output, 
            activation=output_activation_func, 
            kernel_initializer=output_kernel_initializer)
        
        
    @tf.function
    def call(self, inputs, training=None):
        '''
        Расчёт значений модели

        Parameters
        ----------
        inputs : TYPE
            Входные данные сети(состояние).
        training : TYPE
            режим тренировки.

        Returns
        -------
        output : TYPE
            DESCRIPTION.

        '''
        x = self.input_layer(inputs)
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
        x = self.output_layer(x)
        return x
    
    def get_config(self):
        config = super(LSTMNeuralNetworkModel, self).get_config() 
        config.update({'num_input': self.num_input,
                       'lstm_units': self.lstm_units,
                       'hidden_units': self.hidden_units,
                       'num_output': self.num_output,
                       'timesteps': self.timesteps,
                       'activation_func': self.activation_func,
                       'kernel_initializer': self.kernel_initializer,
                       'output_activation_func': self.output_activation_func,
                       'output_kernel_initializer': self.output_kernel_initializer})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
class RecurentNeuralNetworkLayerBlock(tf.keras.layers.Layer):

    def __init__(self,
                 units,
                 timesteps,
                 **kwargs
                 ):
      
        super(RecurentNeuralNetworkLayerBlock, self).__init__(**kwargs)
        
        self.units = units
        self.timesteps = timesteps
        
        self.lstm_layer = tf.keras.layers.LSTM(units)

    @tf.function
    def call(self, inputs, training=None):
      
        x = self.lstm_layer(inputs)
        return x
    
    def get_config(self):
        
        config = super(RecurentNeuralNetworkLayerBlock, self).get_config()
        config.update({ 'units': self.units,
                        'timesteps': self.timesteps})
        
        return config
    
class SimpleNeuralNetworkLayerBlock(tf.keras.layers.Layer):

    def __init__(self,
                 units,
                 activation_func='tanh',
                 kernel_initializer='RandomNormal',
                 **kwargs
                 ):
      
        super(SimpleNeuralNetworkLayerBlock, self).__init__(**kwargs)
        
        self.units = units
        self.activation_func = activation_func
        self.kernel_initializer = kernel_initializer
        
        self.dense_layer = tf.keras.layers.Dense(
            units, 
            kernel_initializer=kernel_initializer)
        self.norm_layer = tf.keras.layers.BatchNormalization()
        self.activation_layer = tf.keras.activations.get(activation_func)

    @tf.function
    def call(self, inputs, training=None):
      
        x = self.dense_layer(inputs)
        x = self.norm_layer(x, training)
        x = self.activation_layer(x)
        return x
    
    def get_config(self):
        
        config = super(SimpleNeuralNetworkLayerBlock, self).get_config()
        config.update({ 'units': self.units,
                        'activation_func': self.activation_func,
                        'kernel_initializer': self.kernel_initializer})
        
        return config
