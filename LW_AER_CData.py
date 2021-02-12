from matplotlib import pylab as plt
from attention import attention
import math
import scipy.io as sio
import numpy as np
import tensorflow as tf
import random
import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING) #DEBUG, INFO, WARNING, ERROR
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def multilayer_perceptron(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

### DFE 参数 ###
hidden_n = 50 # LSTM隐层节点个数
learning_rate = 0.001 #固定学习率
n_epochs = 1 #训练周期数
hidden_mlp = 20  # MLP隐层节点个数
layer_n = 1   # LSTM层数
n_steps = 40 #序列长度
AttentionSize = 30
rnd_select_prob = 0.02  # 每个批次随机选择无标签样本比例
const_phy = 0.5  # 权重参数

### LWAER 参数 ###
N_L_F = 50 # 选择最相似的100个有标签样本
N_U_F = 5000 # 选择最相似的2000个有标签样本
n_hidden_1 = 200  # 1st layer num features
n_hidden_2 = 120  # 2nd layer num features
n_hidden_3 = 80  # 3rd layer num features
hidden_mlp_2 = 40
n_epochs_lw = 200 #训练周期数
learning_rate2 = 0.01
rnd_sp = 1  # 每个批次随机选择无标签样本比例

CD = sio.loadmat("CD.mat")  # 读取历史数据
CD = CD["CD"]
ind = sio.loadmat("ind.mat") # 历史数据中有标签样本序号
ind = ind["ind"]
Lh = sio.loadmat("Lh.mat") # 历史数据中有标签样本序号
Lh = Lh["Lh"]

N_h_l = Lh.shape[0]  # 历史数据中有标签样本个数
N_hist = np.int(Lh[N_h_l-1])  # 历史数据样本长度
n_inputs = CD.shape[1] - 1  # 输入变量维度
n_output = 1  #Y变量维度

# 数据合并在一起序列化
CData = CD
n_samples0 = CData.shape[0]
n_samples = n_samples0 - n_steps  # 序列样本个数
inda = ind - np.ones([n_samples, 1])
inda = inda.reshape(n_samples).astype(int)
xs = []
ys = []
for i in range(n_samples):
    k = i + n_steps
    x0 = [CData[k - j][0:n_inputs] for j in range(n_steps, 0, -1)]
    y0 = [CData[k][-1]]
    xs.append(x0)
    ys.append(y0)
CDTX = np.array(xs)
CDTY = np.array(ys)
CDTX = CDTX[inda]
CDTY = CDTY[inda]

CDS_hx = CDTX[0: N_hist]
CDS_ox = CDTX[N_hist:n_samples]
CDS_hy = CDTY[0: N_hist]
CDS_oy = CDTY[N_hist:n_samples]
N_online = CDS_oy.shape[0]  # 在线数据样本长度

L_h = Lh - np.ones([Lh.shape[0], 1])  # 有标签样本序号
L_h = L_h.reshape(N_h_l).astype(int)
ind_h = np.arange(0, N_hist - n_steps)
U_h = list(set(ind_h).difference(set(L_h))) #剩下的样本全部作为无标签样本
CDS_hx_l = CDS_hx[L_h]  # 历史数据中有标签样本集
CDS_hy_l = CDS_hy[L_h]
CDS_hx_u = CDS_hx[U_h]  # 历史数据中无标签样本集

N_h_u = len(U_h)  # 无标签样本个数
u_h_ind = np.arange(N_h_u)
np.random.shuffle(u_h_ind)
ul_size = N_h_u  # 选取的无标签样本个数
Uh_ind = u_h_ind[0: ul_size] #只取前ul_size个无标签样本
CDS_hx_u = CDS_hx_u[Uh_ind]  # 选取的历史数据中无标签样本集

CD_Lx = CDS_hx_l # 建立有标签样本库，把历史有标签样本放进去，后续每次遇到新的有标签样本都会添加进来
CD_Ly = CDS_hy_l
CD_U = CDS_hx_u  # 建立无标签样本库，把历史无标签样本放进去，后续在线遇到的无标签样本也都添加进来
yp = np.empty(shape=[0, 1])  # 建立空的预测值集合
x = np.empty(shape=[0, n_inputs]) # 建立空的测试集输入集合
y = np.empty(shape=[0, 1])  # 建立空的测试集输出真实值集合
WL = np.empty(shape=[N_L_F, 0])
WU = np.empty(shape=[N_U_F, 0])

for n in range (N_online):
    print(n+1)
    if CDS_oy[n] == -2.0:  # 针对第n个在线样本，判断其y值是否为-2
        CD_U = np.concatenate([CD_U, CDS_ox[n].reshape(1, n_steps, n_inputs)],axis=0)  # 如果是-2，就把这个样本存到在线无标签样本集里
    else: # 如果不是-2，说明遇到有标签样本了，利用已有的有标签和无标签样本建立局部预测模型：
        # print(n+1)
        tf.reset_default_graph()  # 用于清除默认图形堆栈并重置全局默认图形
        N_U = CD_U.shape[0]  # 无标签样本库的长度
        N_L = CD_Ly.shape[0]  # 有标签样本库的长度
        n_batch = np.int(1 / rnd_select_prob)  # 批次数目
        batch_size = np.int((N_U) * rnd_select_prob)  # 批次样本大小

        # 构建LSTM_AE
        input_data = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
        with tf.variable_scope('encoder'):
            lstm_units_enc = [tf.nn.rnn_cell.LSTMCell(hidden_n, use_peepholes=True) for _ in range(layer_n)]
            multi_lstm_enc = tf.nn.rnn_cell.MultiRNNCell(lstm_units_enc)
            # DROPOUT non-recurrent connections of LSTM cells
            # encoder_cell = tf.nn.rnn_cell.DropoutWrapper(encoder_cell, output_keep_prob=0.85)
            encoder_output, encoder_state = tf.nn.dynamic_rnn(multi_lstm_enc, input_data, dtype=tf.float32)
        # Attention layer
        with tf.name_scope('Attention_layer'):
            attention_output, alphas = attention(encoder_output, AttentionSize, return_alphas=True)
        decoder_input1 = tf.tile(attention_output, [1, n_steps])  # 复制然后重组，形成序列
        decoder_input = tf.reshape(decoder_input1, [-1, n_steps, hidden_n])
        with tf.variable_scope('decoder'):
            lstm_units_dec = [tf.nn.rnn_cell.LSTMCell(hidden_n, use_peepholes=True) for _ in range(layer_n)]
            multi_lstm_dec = tf.nn.rnn_cell.MultiRNNCell(lstm_units_dec)
            # DROPOUT non-recurrent connections of LSTM cells
            # decoder_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_cell, output_keep_prob=0.85)
            decoder_output, decoder_state = tf.nn.dynamic_rnn(multi_lstm_dec, inputs=decoder_input, initial_state=encoder_state)
        inference = tf.layers.dense(inputs=decoder_output, units=n_inputs)
        # construct error of Y for supervision
        yy = tf.placeholder("float", [None, n_output], name='Y')
        weights = {
            'h1': tf.Variable(tf.truncated_normal([hidden_n, hidden_mlp])),
            'out': tf.Variable(tf.truncated_normal([hidden_mlp, n_output]))
        }
        biases = {
            'b1': tf.Variable(tf.truncated_normal([hidden_mlp])),
            'out': tf.Variable(tf.truncated_normal([n_output]))
        }
        ypred = multilayer_perceptron(attention_output, weights, biases)
        with tf.variable_scope('loss'):
            # Loss is reconstruction error and prediction error
            # loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(input_data, inference)))) + tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(yy, ypred[0:N_L]))))
            loss = tf.reduce_sum(tf.square(tf.subtract(input_data, inference))) +\
                   tf.reduce_sum(tf.square(tf.subtract(yy, ypred[0:N_L])))
        with tf.variable_scope('training'):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False)
            training = optimizer.minimize(loss, global_step=global_step)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # 增量分配显存
        # 开启session
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        loss_tr_rec = []
        for ep in range(n_epochs):
            ind1 = np.arange(N_U)
            np.random.shuffle(ind1)
            tr_U = CD_U[ind1]
            tr_Lx = CD_Lx
            tr_Ly = CD_Ly
            total_loss_tr = 0.
            for ei in range(n_batch):
                end = (ei + 1) * batch_size
                begin = end - batch_size
                x_ul_value = tr_U[begin : end]
                x_l_value = tr_Lx
                y_l_value = tr_Ly
                x_value = np.concatenate((x_l_value, x_ul_value), axis=0)
                y_value = y_l_value
                _, loss_tr = sess.run([training, loss], feed_dict={input_data: x_value, yy: y_value})
                total_loss_tr += (loss_tr / batch_size)
            avg_loss = total_loss_tr / n_batch
            loss_tr_rec.append(avg_loss)
            print('Epoch %s' % (ep + 1), "avgTrainingloss=", "{:.9f}".format(avg_loss))

########## 动态特征与原始特征组合 ###############
        teX = CDS_ox[n].reshape(1, n_steps, n_inputs)  # 测试数据
        teY = CDS_oy[n]
        # 提取训练数据的动态特征
        CD_X = np.concatenate((CD_Lx, CD_U), axis=0)
        DF_tr = sess.run(attention_output, feed_dict={input_data: CD_X})
        DF_tr_l = DF_tr[0: CD_Lx.shape[0]]  # 有标签样本对应的动态特征
        DF_tr_u = DF_tr[CD_Lx.shape[0] : N_U+N_L]  # 无标签样本对应的动态特征
        # 提取测试数据的动态特征
        DF_ts = sess.run(attention_output, feed_dict={input_data: teX})

        CDO_Lx = CD_Lx[:, -1, :]
        CDO_Ly = CD_Ly
        CDO_U = CD_U[:, -1, :]
        teXO = CDS_ox[n, -1, :].reshape(1, n_inputs)
        teYO = CDS_oy[n]

        ##### 原始特征与动态特征组合
        # CDC_Lx = np.hstack([CDO_Lx, DF_tr_l])  ##特征组合，有标签样本部分
        # CDC_Ly = CDO_Ly
        # CDC_Ux = np.hstack([CDO_U, DF_tr_u])  ## 无标签样本部分
        # CDCt = np.hstack([teXO, DF_ts])  # 测试数据部分

        ##### 只保留原始特征
        CDC_Lx = CDO_Lx  ##特征组合，有标签样本部分
        CDC_Ly = CDO_Ly
        CDC_Ux = CDO_U  ## 无标签样本部分
        CDCt = teXO  # 测试数据部分

        N_inputs = CDC_Lx.shape[1]  # 组合后特征维度

########## 特征样本之间距离和权重计算 ##############################
        Dist_F_l = np.zeros([CDC_Lx.shape[0], CDCt.shape[0]]) # 建立空的有标签样本特征距离矩阵
        for ii in range(CDC_Lx.shape[0]): # 针对第 ii 个有标签样本特征
            ax1_l = (CDC_Lx[ii] - CDCt)
            ax2_l = np.transpose(CDC_Lx[ii] - CDCt)
            Dist_F_l[ii] = np.sqrt(np.dot(ax1_l, ax2_l))
        Ind_F_l = np.argsort(Dist_F_l[:, 0]) #将计算得到的欧氏距离按照从小到大排序，得到序号索引放在 Ind_F_l
        S_F_l = Dist_F_l[Ind_F_l] # 把距离按照索引进行排序
        Ind_ls = Ind_F_l[0:N_L_F]  # 取出前N_L_F个索引
        S_F_Ls = S_F_l[0:N_L_F]  # 取出对应的距离
        CDC_Lxs = CDC_Lx[Ind_ls] # 取出对应的N_L_F个无标签特征
        CDC_Lys = CDC_Ly[Ind_ls]
        sigma_l = np.std(S_F_Ls)  # 距离的标准差
        W_F_l = np.exp((-1) * (S_F_Ls) / (sigma_l * const_phy)) # 把距离值转化成权重
        W_F_L0 = np.tile(W_F_l, (1, N_inputs))
        W_F_L = W_F_L0.reshape(-1, N_inputs)   # 有标签样本权重矩阵

        Dist_F_u = np.zeros([CDC_Ux.shape[0], CDCt.shape[0]]) # 建立空的有标签样本特征距离矩阵
        for ii in range(CDC_Ux.shape[0]): # 针对第 ii 个无标签样本特征
            ax1_u = (CDC_Ux[ii] - CDCt)
            ax2_u = np.transpose(CDC_Ux[ii] - CDCt)
            Dist_F_u[ii] = np.sqrt(np.dot(ax1_u, ax2_u))
        Ind_F_u = np.argsort(Dist_F_u[:, 0]) #将计算得到的欧氏距离按照从小到大排序，得到序号索引放在 Ind_F_u
        S_F_u = Dist_F_u[Ind_F_u] # 把距离按照索引进行排序
        Ind_us = Ind_F_u[0:N_U_F]  # 取出前N_U_F个索引
        S_F_Us = S_F_u[0:N_U_F]  # 取出对应的距离
        CDC_Uxs = CDC_Ux[Ind_us]  # 取出对应的N_U_F个无标签特征
        sigma_u = np.std(S_F_Us)  # 距离的标准差
        W_F_u = np.exp((-1) * (S_F_Us) / (sigma_u * const_phy)) # 把距离值转化成权重
        W_F_U0 = np.tile(W_F_u, (1, N_inputs))
        W_F_U = W_F_U0.reshape(-1, N_inputs)  # 无标签样本权重矩阵

######### 构建LW模型 ##########
        tf.reset_default_graph()  # 用于清除默认图形堆栈并重置全局默认图形
        N_Us = CDC_Uxs.shape[0]  # 无标签样本库的长度
        N_Ls = CDC_Lxs.shape[0]  # 有标签样本库的长度
        n_batchs = np.int(1 / rnd_sp)  # 批次数目
        batch_sizes = np.int((N_Us) * rnd_sp)  # 批次样本大小

        # 构建LW_SAENN
        w_l = tf.placeholder(tf.float32, [None, N_inputs])
        w_u = tf.placeholder(tf.float32, [None, N_inputs])
        X = tf.placeholder(tf.float32, [None, N_inputs])
        yys = tf.placeholder("float", [None, n_output], name='Ys')
        weights = {
            'encoder_h1': tf.Variable(tf.truncated_normal([N_inputs, n_hidden_1])),
            'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
            'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
            'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2])),
            'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1])),
            'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_1, N_inputs])),
        }
        weights_nn = {
            'h1': tf.Variable(tf.truncated_normal([n_hidden_3, hidden_mlp_2])),
            'out': tf.Variable(tf.truncated_normal([hidden_mlp_2, n_output])),
        }
        biases = {
            'encoder_b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
            'encoder_b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
            'encoder_b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
            'decoder_b1': tf.Variable(tf.truncated_normal([n_hidden_2])),
            'decoder_b2': tf.Variable(tf.truncated_normal([n_hidden_1])),
            'decoder_b3': tf.Variable(tf.truncated_normal([N_inputs])),
        }
        biases_nn = {
            'b1': tf.Variable(tf.truncated_normal([hidden_mlp_2])),
            'out': tf.Variable(tf.truncated_normal([n_output])),
        }
        def encoder(x):
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
            layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
            return layer_3
        def decoder(x):
            layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
            layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
            layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))
            return layer_3
        encoder_op = encoder(X)
        decoder_op = decoder(encoder_op)
        ypreds = multilayer_perceptron(encoder_op, weights_nn, biases_nn)

        loss_s = tf.reduce_sum(tf.multiply(tf.square(tf.subtract(X[N_Ls: N_Ls + batch_sizes], decoder_op[N_Ls:N_Ls+batch_sizes])), w_u)) + \
                 tf.reduce_sum(tf.multiply(tf.square(tf.subtract(X[0:N_Ls], decoder_op[0:N_Ls])), w_l)) +\
                 tf.reduce_sum(tf.multiply(tf.square(tf.subtract(yys, ypreds[0:N_Ls])), w_l[:, 0]))
        Tr_opt = tf.train.AdamOptimizer(learning_rate2).minimize(loss_s)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # 增量分配显存
        # 开启session
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())
        loss_tr_rec = []
        for ep in range(n_epochs_lw):
            ind1s = np.arange(N_Us)
            np.random.shuffle(ind1s)
            tr_U = CDC_Uxs[ind1s]
            trW_U = W_F_U[ind1s]
            tr_Lx = CDC_Lxs
            tr_Ly = CDC_Lys
            total_loss_tr = 0.
            for ei in range(n_batchs):
                end = (ei + 1) * batch_sizes
                begin = end - batch_sizes
                x_ul_value = tr_U[begin : end]
                w_u_value = trW_U[begin : end]
                x_l_value = tr_Lx
                y_l_value = tr_Ly
                x_value = np.concatenate((x_l_value, x_ul_value), axis=0)
                y_value = y_l_value
                _, loss_tr = sess.run([Tr_opt, loss_s], feed_dict={X: x_value, yys: y_value, w_u: w_u_value, w_l: W_F_L})
                total_loss_tr += (loss_tr / batch_sizes)
            avg_loss = total_loss_tr / n_batchs
            loss_tr_rec.append(avg_loss)
            print('LWEpoch %s' % (ep + 1), "avgLWTrainingloss=", "{:.9f}".format(avg_loss))

########## Prediction  ##############
        QQ = sess.run(ypreds, feed_dict={X: CDCt})  #  利用模型预测输出

        if math.isnan(QQ):  # 如果这个打标值不小心变成了nan
            QQ = np.mean(CD_Ly)  # 就用标签值的均值打标
        print(np.abs(QQ - teY))
        yp = np.append(yp, QQ.reshape(1,1), axis=0)  # 预测值放到预测集里面

        x = np.append(x, CDS_ox[n, -1, :].reshape(1, n_inputs), axis=0)  # 测试集输入放到集合里面
        y = np.append(y, CDS_oy[n].reshape(1,1), axis=0)  # 测试集真实输出放到集合里面

        CD_Lx = np.concatenate((CD_Lx, CDS_ox[n].reshape(1, n_steps, n_inputs)), axis=0)  # 把当前遇到的有标签样本添加到有标签样本库中
        CD_Ly = np.concatenate((CD_Ly, CDS_oy[n].reshape(1, n_output)), axis=0)
        WL = np.concatenate((WL, W_F_l), axis=1)  # 存住权重向量
        WU = np.concatenate((WU, W_F_u), axis=1)
        sio.savemat('Result_LW_AER_W.mat', {'x': x, 'y': y, 'yp': yp, 'WL': WL, 'WU': WU})  # 保存需要的东西
# 画图
yo = y
yp = yp
s = range(len(x))
fig,ax = plt.subplots()
plt.plot(s, yo, 'b.-', linewidth=2, label='Real Value')
plt.plot(s, yp, 'ro-', linewidth=2, label='Predicted Value')
plt.title('Predicted Value')
plt.xlabel('Sample Points')
plt.ylabel('fcao content')
plt.legend(loc= 'best')
plt.show()

# 测试集RMSE计算
RMSE_ts = np.sqrt(np.sum(np.square(yp - yo)) / (len(x)))
print("Testing RMSE:", "{:.9f}".format(RMSE_ts))

#测试集R2计算
SStot = np.sum(np.square(yo - np.mean(yo)))
SSres = np.sum(np.square(yo - yp))
R2_ts = 1 - SSres/SStot
print("Testing R2:", "{:.9f}".format(R2_ts))

#测试集计算MAPE
MAPE = np.sum(np.abs(yo - yp) / np.abs(yo)) / (len(x))
print("Testing MAPE:", "{:.9f}".format(MAPE))

#测试集计算MAE
MAE = np.sum(np.abs(yo - yp)) / (len(x))
print("Testing MAE:", "{:.9f}".format(MAE))