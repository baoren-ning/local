import numpy as np
from tensorflow import keras
from sko.GA import GA
from sko.DE import DE
import tensorflow_addons as tfa
from scipy import interpolate
import matplotlib.pyplot as plt
data=np.genfromtxt(r'..\data\x.csv', delimiter=',')
lamda=np.arange(8000,13001,1)*10**-9
T_sur_h=373
T_sur_l=300
inter_x=data[41:96]#cst中出来的x，用来插值
network1 = keras.models.load_model(r'..\NN\real_h_2.h5')
network2 = keras.models.load_model(r'..\NN\imag_h_2.h5')
network3 = keras.models.load_model(r'..\NN\spe.h5')
def I_bb(lamda, T):
    C1 = 3.741 * 10 ** -16 # W.m ^ 2 / sr
    C2 = 0.01438769 # m.K
    result = C1/ ((lamda** 5)* (np.exp(C2 / (lamda * T)) - 1))
    return result

def cal_emittance_rad(lamda,inter_x,inter_y,T_sur):
    BB_Tsur_l=I_bb(lamda,T_sur).reshape(1,-1)
    abs_lamda= interpolate.interp1d(inter_x*10**-6,inter_y)
    ynew=abs_lamda(lamda)
    emittance=ynew
    emittance_rad=np.trapz(BB_Tsur_l*emittance,lamda)/np.trapz(BB_Tsur_l,lamda)
    return emittance_rad

def schaffer(pa):
    p, l, t1, t2 = pa
    data_r = np.array([p,l, t1, t2])
    out_real_h = network1(data_r.reshape(1, 4))
    out_imag_h = network2(data_r.reshape(1, 4))
    spectrum_h = np.sqrt(out_real_h ** 2 + out_imag_h ** 2)
    # out_real_l = network3(data_r.reshape(1, 4))
    # out_imag_l = network4(data_r.reshape(1, 4))
    # spectrum_l = np.sqrt(out_real_l ** 2 + out_imag_l ** 2)
    # if spectrum_l.max()>1 or spectrum_l.min()<0:
    #     spectrum_l = np.zeros((spectrum_l.shape))
    spectrum_l = network3(data_r.reshape(1, 4))
    inter_y_h = 1-spectrum_h[:, 41:96]*spectrum_h[:, 41:96]
    inter_y_l = 1 - spectrum_l[:, 41:96] * spectrum_l[:, 41:96]
    em_h = cal_emittance_rad(lamda, inter_x, inter_y_h, T_sur_h)
    em_l = cal_emittance_rad(lamda, inter_x, inter_y_l, T_sur_l)
    deta=em_h-em_l
    return float(1-deta)
constraint_ueq = [
    lambda x: 0.1 - x[0] + x[1],
]

ga = DE(func=schaffer, n_dim=4, size_pop=100, max_iter=100, prob_mut=0.001, lb=[0.6,0.24, 0.6, 0.1],
            ub=[2,1.8,2, 0.4], constraint_ueq=constraint_ueq)
# ga = GA(func=schaffer, n_dim=4, size_pop=50, max_iter=100, prob_mut=0.001, lb=[0.6,0.24, 0.6, 0.1],
#             ub=[2,1.8,2, 0.4], precision=1e-2)
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)
