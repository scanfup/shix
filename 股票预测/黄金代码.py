import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
from keras import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import LSTM

df = pd.read_csv('Gold Price (2013-2023).csv')#读取csv文件的方法，读取csv文件后，会返回一个dataframe对象，并将其赋值给变量df
#通过df，你可以对该数据对象进行一系列的操作，以下是一些示例操作
print("原df")
print(df.head())#比如，查看前五行数据
#print(df.columns)#查看数据列的名称
#df.info()#查看数据的一些基本信息
df.drop(['Vol.', 'Change %'], axis=1, inplace=True)#删除对应列，vol和change列，axis=1表示列，axis=0表示行，inplace表示直接在元df上进行操作，在源df上进行修改，指定的列将被删除
df['Date'] = pd.to_datetime(df['Date']) #使用datetime函数将df中的date列转化为datetime对象，可以更方便的对日期进行排序，
df.sort_values(by='Date', ascending=True, inplace=True)#使用排序功能按date列对数据进行升序排序(从过去到现在)ascending指定升序排列，inplace表示在源df上进行操作，而不是返回一个新的df
df.reset_index(drop=True, inplace=True)#这行代码使用 reset_index() 函数重置 DataFrame 的索引。drop=True 表示删除原索引，而不是将原索引作为新列保留，inplace=True 表示直接在原 DataFrame 上进行操作。这样做的结果是，df 将有一个新的、从0开始的连续索引。
#这些步骤的目的通常是为了准备数据，以便进行进一步的分析或可视化。通过将日期转换为日期时间格式并按日期排序，你可以确保数据按照时间顺序排列，这对于时间序列分析非常重要。重置索引可以使数据更加整洁，便于访问和处理。

NumCols = df.columns.drop(['Date'])#这行代码从 df 的列名中排除 Date 列，得到一个只包含数值列（假设其他列都是数值列）的列表，并将其赋值给 NumCols 变量。
df[NumCols] = df[NumCols].replace({',': ''}, regex=True)#这行代码使用 replace() 函数将 NumCols 列中的逗号替换为空字符串。参数 {',': ''} 表示将逗号替换为空字符串，regex=True 表示使用正则表达式。这样做是为了将带有逗号的数值转换为适合数值类型的格式。
df[NumCols] = df[NumCols].astype('float64')#将数值列转化为浮点数
print("更改后:")
print(df.head())
df.head()#修改后，查看前五行数据
# 计算重复行的数量
df.duplicated().sum()

# 计算所有缺失值的数量
df.isnull().sum().sum()

# 使用Plotly创建价格随日期变化的折线图
fig = px.line(y=df.Price, x=df.Date)

# 更新折线图的线条颜色为黑色
fig.update_traces(line_color='black')

# 更新图表的布局，包括坐标轴标题和图表标题
fig.update_layout(xaxis_title="Date",
                  yaxis_title="Scaled Price",
                  title={'text': "Gold Price History Data", 'y':0.95, 'x':0.5, 'xanchor':'center', 'yanchor':'top'},
                  plot_bgcolor='rgba(255,223,0,0.8)')

# 计算2022年的数据行数，作为测试集的大小
test_size = df[df.Date.dt.year == 2022].shape[0]


# 设置绘图的大小和分辨率
plt.figure(figsize=(10, 4), dpi=150)

# 设置绘图的背景颜色为黄色
plt.rcParams['axes.facecolor'] = 'yellow'
plt.rc('axes', edgecolor='white')

# 绘制训练集数据的折线图（黑色）
plt.plot(df.Date[:-test_size], df.Price[:-test_size], color='black', lw=2)

# 绘制测试集数据的折线图（蓝色）
plt.plot(df.Date[-test_size:], df.Price[-test_size:], color='blue', lw=2)

# 设置图表标题和坐标轴标签
plt.title('Gold Price Training and Test Sets', fontsize=15)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)

# 添加图例
plt.legend(['Training set', 'Test set'], loc='upper left', prop={'size': 15})

# 设置网格颜色
plt.grid(color='white')

# 显示图表
plt.show()

# 创建并拟合MinMaxScaler，用于将价格数据缩放到[0,1]范围
scaler = MinMaxScaler()
scaler.fit(df.Price.values.reshape(-1, 1))

# 设置窗口大小为60
window_size = 60

# 获取训练集数据（不包含测试集数据）
train_data = df.Price[:-test_size]

# 将训练集数据缩放到[0,1]范围
train_data = scaler.transform(train_data.values.reshape(-1, 1))
#这里使用 MinMaxScaler 将训练集数据缩放到 [0,1] 范围。缩放数据的目的是为了让模型训练更稳定、收敛更快。train_data.values.reshape(-1, 1) 将 train_data
# 从一维数组（例如 [1500, 1520, 1480, ...]）转换为二维数组（例如 [[1500], [1520], [1480], ...]），以适应 scaler.transform 的输入格式。
# 初始化训练集的输入（X_train）和输出（y_train）列表
X_train = []
y_train = []

# 使用滚动窗口创建训练集的输入和输出数据
for i in range(window_size, len(train_data)):
    X_train.append(train_data[i - window_size:i, 0])
    y_train.append(train_data[i, 0])
#抱歉，我之前给出的解释有些混淆。在时间序列预测中，通常使用滚动窗口的方法来创建训练集的输入和输出数据。这种方法可以有效地利用历史数据来训练模型，以预测未来的值。

# 让我更清楚地解释一下滚动窗口方法在这里的应用：
# 滚动窗口方法
# 在时间序列预测中，我们希望使用过去的一段时间窗口内的数据来预测未来的值。这里的滚动窗口指的是我们从时间序列数据中滑动的一段固定长度的窗口，每次移动一个时间步（或几个时间步），以创建训练集的样本。
# 具体步骤
# 选择窗口大小（window_size）：定义每个训练样本包含的时间步数。例如，如果窗口大小是3，则每个训练样本将包含连续的3个时间步的数据。
# 创建训练集的输入和输出：
# 输入（X_train）：每个训练样本的输入是一个长度为 window_size 的时间窗口，包含连续的 window_size 个时间步的数据。
# 输出（y_train）：对应每个训练样本的输出是该时间窗口之后的一个时间步的数据，即我们希望模型预测的目标值。
# 滚动创建样本：从时间序列数据的开头开始，每次移动一个时间步，将当前窗口内的数据添加到 X_train，并将下一个时间步的数据添加到 y_train。
#在你的代码中，使用了类似滚动窗口的方法来创建训练集的输入和输出数据。具体来说，这里的循环从 window_size 开始，每次循环取出当前时间窗口内的数据作为一个训练样本的输入，同时取出该时间窗口之后的一个时间步的数据作为对应的输出。
# 希望这次的解释更加清晰明了！如果还有疑问或需要进一步帮助，请随时告诉我。

# 获取包含测试集数据和前60天数据的集合
test_data = df.Price[-test_size - window_size:]

# 将测试集数据缩放到[0,1]范围
test_data = scaler.transform(test_data.values.reshape(-1, 1))

# 初始化测试集的输入（X_test）和输出（y_test）列表
X_test = []
y_test = []

# 使用滚动窗口创建测试集的输入和输出数据
for i in range(window_size, len(test_data)):
    X_test.append(test_data[i - window_size:i, 0])
    y_test.append(test_data[i, 0])
#在机器学习和深度学习中，通常会将数据集划分为训练集（Training Set）和测试集（Test Set），它们各自有不同的作用和使用方式：

# 训练集（Training Set）
# 训练集是用来训练机器学习模型的数据集。模型通过训练集中的数据来学习特征之间的关系、模式和规律。在训练过程中，模型会根据训练集的输入和对应的目标输出（标签）进行优化和调整，以尽可能准确地预测目标变量。
#
# 测试集（Test Set）
# 测试集是用来评估模型泛化能力的数据集。在模型训练完成后，我们需要通过测试集来评估模型在未见过的数据上的表现。测试集中的数据没有参与模型的训练过程，因此可以用来模拟模型在真实场景中的表现。
#
# 区别与作用
# 数据来源：
#
# 训练集：用于训练模型的数据，包含输入特征和对应的目标输出。
# 测试集：独立于训练集的数据，用于评估模型在未知数据上的预测能力。
# 使用方式：
#
# 训练集：用来调整模型参数，使其能够对训练数据中的模式进行拟合。
# 测试集：用来评估模型的泛化能力，即模型在未知数据上的表现。
# 防止过拟合：
#
# 通过在训练集上训练模型，然后在测试集上评估模型性能，可以帮助我们避免过拟合（Overfitting）。过拟合指的是模型在训练集上表现良好，但在未知数据上表现较差的情况。
# 在你的代码中的应用
# 在你的代码中，使用了 test_size 来划分测试集。具体来说：
#
# 训练集数据：train_data 是通过从完整数据集中减去测试集数据（即最后 test_size 行数据）而得到的。
# 测试集数据：test_data 是包含最后 test_size 行数据的子集，用于最后评估模型在这部分数据上的预测性能。
# 通过这种方式，你可以确保模型在训练过程中不会接触到测试集数据，从而更准确地评估模型的泛化能力。
#
# 希望这个解释能够帮助你理解训练集和测试集的区别及其在机器学习中的重要性。如果还有其他问题或需要进一步帮助，请随时告诉我！
# 将训练集和测试集的数据转换为NumPy数组
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# 调整训练集和测试集输入数据的形状以适应LSTM模型
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 调整训练集和测试集输出数据的形状
y_train = np.reshape(y_train, (-1, 1))
y_test = np.reshape(y_test, (-1, 1))

# 打印训练集和测试集数据的形状
print('X_train Shape: ', X_train.shape)
print('y_train Shape: ', y_train.shape)
print('X_test Shape:  ', X_test.shape)
print('y_test Shape:  ', y_test.shape)

# 定义LSTM模型
# LSTM（长短期记忆网络，Long Short-Term Memory）是一种特殊类型的循环神经网络（RNN），专门设计用来处理和预测时间序列数据。它是由Hochreiter和Schmidhuber在1997年提出的，主要解决了普通RNN在长序列数据上面临的梯度消失或梯度爆炸的问题。
#
# LSTM的特点和优势包括：
# 长期记忆能力：LSTM通过门控结构（输入门、遗忘门和输出门）来控制信息的流动，能够有效地捕捉和记忆长期依赖关系，适用于长序列数据的建模。
#
# 防止梯度消失问题：通过门控机制，LSTM可以有效地防止在反向传播过程中梯度消失或梯度爆炸的问题，从而更稳定和有效地训练模型。
#
# 适应多种序列预测任务：由于其灵活的结构和能力，LSTM广泛应用于语音识别、自然语言处理、时间序列预测等领域。
#
# LSTM模型的结构和工作原理：
# 输入门（Input Gate）：决定新信息是否进入细胞状态。
# 遗忘门（Forget Gate）：决定哪些信息从细胞状态中删除。
# 输出门（Output Gate）：决定细胞状态的哪部分输出给下一层或输出层。
# LSTM在时间序列预测中的应用：
# 在你的代码中，使用LSTM来建模金价时间序列数据。通过多层LSTM结构，模型可以学习复杂的时间依赖关系，从而进行准确的价格预测。在训练过程中，通过优化损失函数来调整模型参数，使其能够最小化预测值与真实值之间的误差。
#
# 如果你有具体关于LSTM或者代码中实现的问题，欢迎随时提问，我会尽力帮助解答！
def define_model():
    input1 = Input(shape=(window_size, 1))
    x = LSTM(units=64, return_sequences=True)(input1)
    x = Dropout(0.2)(x)
    x = LSTM(units=64, return_sequences=True)(x)
    x = Dropout(0.2)(x)
    x = LSTM(units=64)(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='softmax')(x)
    dnn_output = Dense(1)(x)

    # 创建和编译模型
    model = Model(inputs=input1, outputs=[dnn_output])
    model.compile(loss='mean_squared_error', optimizer='Nadam')
    model.summary()

    return model

# 初始化和训练模型
model = define_model()
history = model.fit(X_train, y_train, epochs=150, batch_size=32, validation_split=0.1, verbose=1)
#执行model.fit时打印出来数据训练过程
# 评估模型在测试集上的表现
result = model.evaluate(X_test, y_test)

# 使用模型预测测试集
y_pred = model.predict(X_test)

# 计算平均绝对百分比误差(MAPE)
MAPE = mean_absolute_percentage_error(y_test, y_pred)

# 计算预测准确度
Accuracy = 1 - MAPE

# 打印测试损失、MAPE和准确度
print("Test Loss:", result)
print("Test MAPE:", MAPE)
print("Test Accuracy:", Accuracy)

# 逆缩放测试集的真实值和预测值
y_test_true = scaler.inverse_transform(y_test)
y_test_pred = scaler.inverse_transform(y_pred)

# 绘制模型表现的折线图
plt.figure(figsize=(10, 4), dpi=150)
plt.rcParams['axes.facecolor'] = 'yellow'
plt.rc('axes', edgecolor='white')

# 绘制训练集数据（黑色）
plt.plot(df['Date'].iloc[:-test_size], scaler.inverse_transform(train_data), color='black', lw=2)

# 绘制测试集真实值（蓝色）
plt.plot(df['Date'].iloc[-test_size:], y_test_true, color='blue', lw=2)

# 绘制测试集预测值（红色）
plt.plot(df['Date'].iloc[-test_size:], y_test_pred, color='red', lw=2)

# 设置图表标题和坐标轴标签
plt.title('Model Performance on Gold Price Prediction', fontsize=15)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price', fontsize=12)

# 添加图例
plt.legend(['Training Data', 'Actual Test Data', 'Predicted Test Data'], loc='upper left', prop={'size': 15})

# 设置网格颜色
plt.grid(color='white')

# 显示图表
plt.show()
