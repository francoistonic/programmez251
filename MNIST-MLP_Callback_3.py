# -*- coding: utf-8 -*-

# Importation du module tensorFlow
import tensorflow as tf

# Chargement des données d'apprentissage et de tests
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(60000, 784)
x_test  = x_test.reshape(10000, 784)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test  = tf.keras.utils.to_categorical(y_test, 10)

# Création d'un réseau multicouches
MonReseau = tf.keras.Sequential()
MonReseau.add(tf.keras.layers.Dense(units=50,
                                    input_shape=(784,),
                                    activation='relu'))
MonReseau.add(tf.keras.layers.Dense(units=10,
                                    activation='softmax'))

# COMPILATION du réseau
MonReseau.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# DERIVATION de la classe abstraite
class MaClasseCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    if(logs.get('val_accuracy') > 0.975):
      print("\nARRET DE L'APPRENTISSAGE: 97,5% atteint en validation !")
      self.model.stop_training = True

# INSTANCIATION de la classe MaClasseCallback()
MonCallback = MaClasseCallback()

# APPRENTISSAGE du réseau
hist=MonReseau.fit(x=x_train,
                   y=y_train,
                   epochs=15,
                   verbose=2,
                   batch_size=32,
                   validation_data=(x_test,y_test),
                   callbacks=[MonCallback])

# PERFORMANCES du réseau mémorisé
print("\nEvaluation sur la base de tests:")
perf=MonReseau.evaluate(x=x_test,y=y_test,
                        verbose=2)
print("Résultats du test: {:.2f}%".format(perf[1]*100))