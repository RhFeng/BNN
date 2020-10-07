import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.ticker as ticker
from keras import callbacks
plt.rcParams["font.family"] = "Times New Roman"

#%%
# load the data
# CAL CNC GR HRD HRM PE ZDEN

df1 = pd.read_csv('train.csv')

df1.replace(['-999', -999], np.nan, inplace=True)
df1.dropna(axis=0, inplace=True)


# seperate the features and targets
df1_data = np.array(df1)

x_trainwell = df1_data[6000:7000,:-2]
y_trainwell = df1_data[6000:7000,-2]


features = ['CAL', 'PHI', 'GR', 'DR', 'MR', 'PE', 'RHO']

features_new = [ 'PHI', 'GR', 'DR', 'PE', 'RHO']
n_features = len(features)

index = [1,2,3,5,6]


y_trainwell = np.reshape(y_trainwell,[-1,1])


from plot_well import plot_well_feature, plot_well_target

# define the test location, increment, depth interval
test_loc = [400,700]  #400,700 550,900

test_inc = 100  # 100

dz = 0.1524

tvd = 6000 * dz

plot_well_feature(x_trainwell[:,index], test_loc, test_inc, dz,tvd)



plot_well_target(y_trainwell, test_loc, test_inc, dz,tvd)


#%%
from train_test_split import train_test_split

X_train, Y_train, X_test, Y_test = train_test_split(x_trainwell[:,index], y_trainwell, test_loc, test_inc)

pd_data = pd.DataFrame(data = X_train, columns = features_new)

g = sns.pairplot(pd_data,corner=True,markers="o",
                  plot_kws=dict(s=5, edgecolor="b",  linewidth=1))

g.fig.set_figwidth(8)
g.fig.set_figheight(8)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)



#%%
# standize the matrix for training data
scaler = StandardScaler()
x_trainwell = scaler.fit_transform(x_trainwell)

# split the train and test data

X_train, Y_train, X_test, Y_test = train_test_split(x_trainwell[:,index], y_trainwell, test_loc, test_inc)

ymin = np.min(Y_test) - 5
ymax = np.max(Y_test) + 5
#%%

import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

import tensorflow_probability as tfp

tfd = tfp.distributions

negloglik = lambda y, rv_y: -rv_y.log_prob(y)
#%%
import os
model_name = 'log_csv'
model_dir     = os.path.join('check', model_name)
csv_fn        = os.path.join(model_dir, 'train_log_nn.csv')
csv_logger  = callbacks.CSVLogger(csv_fn, append=True, separator=';')


### Case 1: No Uncertainty
# Build model.
model_nn = tf.keras.Sequential([
   tf.keras.layers.Dense(10,name='Layer_1'),

   tf.keras.layers.Dense(10,name='Layer_2'),

  tf.keras.layers.Dense(1,name='Layer_3'),

])

model_nn._name = 'NN'

# Do inference.
model_nn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.02), loss='mse', metrics=['mse'])
model_nn.fit(X_train, Y_train,batch_size=100, epochs=5000, verbose=1)#, callbacks=[csv_logger])

#%%

Y_test_predict_no = model_nn.predict(X_test)

predict = np.vstack([Y_test_predict_no.ravel(), Y_test.ravel()])

df = pd.DataFrame(np.transpose(predict), columns=["Prediction", "Reference"])

pt = sns.jointplot(x="Prediction", y="Reference", edgecolor="b",data=df,size=3.5,height=20, ratio=5,xlim = (ymin, ymax), ylim = (ymin, ymax))

pt.set_axis_labels("Prediction", "Reference", fontsize=14)

CC = np.corrcoef(Y_test.ravel(), Y_test_predict_no.ravel())

pt.fig.text(0.18, 0.75,'CC = %.4f' %CC[0,1], fontsize=12)
pt.ax_joint.xaxis.set_major_locator(ticker.MultipleLocator(12))
pt.ax_joint.yaxis.set_major_locator(ticker.MultipleLocator(12))



#%%
predict_comp = np.zeros((4,X_test.shape[0]))
predict_comp[0,:] = Y_test_predict_no.ravel()

#%%
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


from keras import backend as K
from keras import activations, initializers
from keras.layers import Layer

import tensorflow as tf
import tensorflow_probability as tfp


class DenseVariational(Layer):
    def __init__(self,
                 units,
                 kl_weight,
                 activation=None,
                 prior_sigma_1=1.5,
                 prior_sigma_2=0.1,
                 prior_pi=0.5, **kwargs):
        self.units = units
        self.kl_weight = kl_weight
        self.activation = activations.get(activation)
        self.prior_sigma_1 = prior_sigma_1
        self.prior_sigma_2 = prior_sigma_2
        self.prior_pi_1 = prior_pi
        self.prior_pi_2 = 1.0 - prior_pi
        self.init_sigma = np.sqrt(self.prior_pi_1 * self.prior_sigma_1 ** 2 +
                                  self.prior_pi_2 * self.prior_sigma_2 ** 2)

        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def build(self, input_shape):
        self.kernel_mu = self.add_weight(name='kernel_mu',
                                         shape=(input_shape[1], self.units),
                                         initializer=initializers.normal(stddev=self.init_sigma),
                                         trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu',
                                       shape=(self.units,),
                                       initializer=initializers.normal(stddev=self.init_sigma),
                                       trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho',
                                          shape=(input_shape[1], self.units),
                                          initializer=initializers.constant(0.0),
                                          trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho',
                                        shape=(self.units,),
                                        initializer=initializers.constant(0.0),
                                        trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        kernel_sigma = tf.math.softplus(self.kernel_rho)
        kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)

        bias_sigma = tf.math.softplus(self.bias_rho)
        bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)

        self.add_loss(self.kl_loss(kernel, self.kernel_mu, kernel_sigma) +
                      self.kl_loss(bias, self.bias_mu, bias_sigma))

        return self.activation(K.dot(inputs, kernel) + bias)

    def kl_loss(self, w, mu, sigma):
        variational_dist = tfp.distributions.Normal(mu, sigma)
        return self.kl_weight * K.sum(variational_dist.log_prob(w) - self.log_prior_prob(w))

    def log_prior_prob(self, w):
        comp_1_dist = tfp.distributions.Normal(0.0, self.prior_sigma_1)
        comp_2_dist = tfp.distributions.Normal(0.0, self.prior_sigma_2)
        return K.log(self.prior_pi_1 * comp_1_dist.prob(w) +
                     self.prior_pi_2 * comp_2_dist.prob(w))
    
from keras.layers import Input
from keras.models import Model

train_size = 800

batch_size = 100
num_batches = train_size / batch_size

kl_weight = 1.0 / num_batches
prior_params = {
    'prior_sigma_1': 0.5, 
    'prior_sigma_2': 2, 
    'prior_pi': 0.2 
}

x_in = Input(shape=(5,))
x = DenseVariational(10, kl_weight, **prior_params, activation='relu',name='Layer_1')(x_in)
x = DenseVariational(10, kl_weight, **prior_params, activation='relu',name='Layer_2')(x)
x = DenseVariational(1, kl_weight, **prior_params,name='Layer_3')(x) 


from keras import optimizers

def neg_log_likelihood(y_obs, y_pred, sigma=2):
    dist = tfp.distributions.Normal(loc=y_pred, scale=sigma)
    return K.sum(-dist.log_prob(y_obs))/10


#%%
csv_fn        = os.path.join(model_dir, 'train_log_bnn_2.csv')
csv_logger  = callbacks.CSVLogger(csv_fn, append=True, separator=';')    



model = Model(x_in, x)   

model._name = 'BNN'
   
model.compile(loss=neg_log_likelihood, optimizer=optimizers.Adam(lr=0.1), metrics=['mse'])
model.fit(X_train, Y_train, batch_size=batch_size, epochs=5000, verbose=1)
#%%
Y_test_predict_no = model.predict(X_test)

predict_comp[3,:] = Y_test_predict_no.ravel()

predict = np.vstack([Y_test_predict_no.ravel(), Y_test.ravel()])

df = pd.DataFrame(np.transpose(predict), columns=["Prediction", "Reference"])

pt = sns.jointplot(x="Prediction", y="Reference", edgecolor="b",data=df,size=3.5,height=20, ratio=5,xlim = (ymin, ymax), ylim = (ymin, ymax))

pt.set_axis_labels("Prediction", "Reference", fontsize=14)

CC = np.corrcoef(Y_test.ravel(), Y_test_predict_no.ravel())

pt.fig.text(0.18, 0.75,'CC = %.4f' %CC[0,1], fontsize=12)
pt.ax_joint.xaxis.set_major_locator(ticker.MultipleLocator(12))
pt.ax_joint.yaxis.set_major_locator(ticker.MultipleLocator(12))


#%%
from plot_well import plot_well_predict

plot_well_predict(Y_test, predict_comp, test_loc, test_inc, dz, tvd)

#%%
def predict(model, data, T):
    
    # predict stochastic dropout model T times
    y_pred_list = []
    
    for t in range(T):
        y_pred = model.predict(data)
        y_pred_list.append(y_pred)
    
    y_preds = np.concatenate(y_pred_list, axis=1)

    y_mean = np.mean(y_preds, axis=1)
    
    y_sigma = np.std(y_preds, axis=1)

    return y_mean, y_sigma

list_stochastic_feed_forwards = [100,200,400,600,800,1000]

result_dict = {}
for ind, num_stochastic_T in enumerate(list_stochastic_feed_forwards):
    print(ind)
    sigma_list = []
    
    prediction, sigma = predict(model, X_test, T=num_stochastic_T)
    sigma_list.append((sigma))
    
    result_dict.update({ '{}'.format(str(num_stochastic_T)) : 
    [num_stochastic_T, 1,
    np.mean(sigma_list), np.std(sigma_list)]} )  
        
#%%
from plot_well import plot_pred_interval        
plot_pred_interval(sigma, Y_test, prediction, test_loc, test_inc, dz, tvd)  
#%%

predict = np.vstack([prediction.ravel(), Y_test.ravel()])

df = pd.DataFrame(np.transpose(predict), columns=["Prediction", "Reference"])

pt = sns.jointplot(x="Prediction", y="Reference", edgecolor="b",data=df,size=3.5,height=20, ratio=5,xlim = (ymin, ymax), ylim = (ymin, ymax))

pt.set_axis_labels("Prediction", "Reference", fontsize=14)

CC = np.corrcoef(Y_test.ravel(), prediction.ravel())

pt.fig.text(0.18, 0.75,'CC = %.4f' %CC[0,1], fontsize=12)
pt.ax_joint.xaxis.set_major_locator(ticker.MultipleLocator(12))
pt.ax_joint.yaxis.set_major_locator(ticker.MultipleLocator(12))





















