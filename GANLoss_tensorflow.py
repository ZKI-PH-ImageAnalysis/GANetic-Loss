import tensorflow

EPSILON = 1e-8

class GANetic(tensorflow.keras.losses.Loss):
    def __init__(
    	self, 
    	eps=1e-8,
      reduction=None,
    	name="ganetic_loss"):
        super().__init__(name=name)
        self.name = name
        self.eps = eps

    def call(self, y_true, y_pred):
        y_pred = tensorflow.keras.backend.clip(y_pred, tensorflow.keras.backend.epsilon(), 1 - tensorflow.keras.backend.epsilon())
        y_true = tensorflow.cast(y_true, tensorflow.float32)

        term_0 = tensorflow.math.pow(y_pred, 3)
      
        term_1_0 = tensorflow.math.divide(y_true, tensorflow.math.add(y_pred, self.eps))
        term_1_1 = tensorflow.math.multiply(3.985, term_1_0)
        term_1 = tensorflow.math.sqrt(tensorflow.math.add(tensorflow.math.abs(term_1_1), self.eps))

        if reduction == "mean":
            loss = tensorflow.reduce_mean(term_0 + term_1)
        else:
            loss = term_0 + term_1
        return loss
