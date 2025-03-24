# 人工智能的编程基础笔记
## 前言：penguin示例的数据处理技术
我们在南极岛上有一帮企鹅，这些企鹅在三个岛上，在科学家的观察下，他们有以下几种数据：
* 我们先用命令打开这个txt看看结果：
```python
def read_data(filename):
    with open(filename) as f:
        lines = f.readlines()
    header = lines[0].strip()
    column_name = header.split(',')
    return column_name

filename = "data\palmer-penguins\palmer-penguins.txt"
column_name = read_data(filename)
print(column_name)
```
这里输出的结果是
```python
['species', 'island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'year']
```
* `strip()` 和 `split()` 是 Python 中常用的字符串处理方法

1. **strip()**:
   - 用于去除字符串开头和结尾的空白字符（包括空格、换行符、制表符等）
   - 不会改变字符串中间的内容
   - 如果不带参数，默认去除空白字符
   - 示例：`"  hello world  ".strip()` 返回 `"hello world"`

2. **split(',')**:
   - 用于将字符串按照指定的分隔符（这里是逗号）拆分成多个子字符串
   - 返回一个包含拆分结果的列表
   - 示例：`"a,b,c".split(',')` 返回 `['a', 'b', 'c']`
* 大概知道有哪几种标签后，我们需要把这些csv里面的数据转成python自己的数据类型——列表：
* 代码实现：
```python
def string2float(str):
    return float(str) if str != '' else None
def read_string(filename, col):
    with open(filename) as f:
        lines = f.readlines()
    item = [line.strip().split(',')[col] for line in lines[1:]]
    return item
def read_float(filename, col):
    with open(filename) as f:
        lines = f.readlines()
    item = [string2float(line.strip().split(',')[col]) for line in lines[1:]]
    return item
species = read_string(filename, col = 0)
island = read_string(filename, col = 1)
bill_length = read_float(filename, col = 2)
bill_depth = read_float(filename, col = 3)
flipper_length = read_float(filename, col = 4)
body_mass = read_float(filename, col = 5)
```
经过这样的代码操作，我们就可以获得结构化的数据，请注意：**这些代码需要读者自行复现才能真正被理解。**

## 回归模型
### 1.线性回归模型1.0版本
* 模型的建立与一步步优化

对于一连串离散的点，如果我们要采用一条直线（线性）来做拟合，我们对这条直线最基本的假设是：有这么一个权重w来加到自变量上，也有这么一个偏置量加到wx后，用数学方式表达就是：

$$ y = wx + b $$
在这个penguin的示例中，我们用flipper_lengnths来预测body_mass,


|| bias | flipper_length | body_mass |
|-|------|----------------|-----------|
|自变量|  1    |    x            |       y    |
|系数|b      | w|y|

我们的核心思路是**用矩阵**规整化这个线性运算——也即把线性运算用矩阵形式做简化表达，进而进一步做研究：
|| bias | flipper_length | body_mass |
|-|------|----------------|-----------|
|自变量|  1    |    x            |       y    |
|系数|w0     | w1|y|

我们可以用矩阵形式表示为：
![picture 0](images/ab05b5060e73831e50f5ad17a9853e94b95fffe860d9fab6162825a7ce8ebb73.png)  
当然，这只是一个样本，我们需要把一个样本扩展到样本集内的$n$个样本，这就需要我们把X的特征维度用列数表示，样本个数用行数表示，对应的y只有一个特征也即（value），但是也有样本个数$n$行。
![picture 2](images/9843b3c6da329ca19e0fbd8848f84276b316d30efa92c37c01b9fefffe2642cf.png)  
现在我们就已经成功把原来$n$个线性表达式用矩阵乘法规整化了，这只是建立模型的第一步，我们还需要有东西来检验模型的好坏，这就需要我们引入评价标准**损失函数**：
我们知道body_mass是有实际值的，而上述的y向量又给到了$n$个样本的预测值，我们可以引入最小二乘法，算这个模型的均方误差MSE
![picture 3](images/b41dadc0f287df835fbe7bd7044b5491bf3e70f720f964b3dacc07c81b6e0506.png)  
对于线性回归模型，我们是可以去解出解析解的：
![picture 4](images/8fcf199f1f8a717f238fbb12461c88504675896e7a64c03b1d566e938cac14a5.png)  
也即：
![picture 5](images/73e5ac3880cb3fb932a25b0583891de6a9f6faf798682ab6ec3b0e3ff92d617c.png)  

**TIPS: 矩阵求导基本规则**：

1. **标量对向量求导**  
   若 y 是标量，x 是向量，则 ∂y/∂x 是一个与 x 同维度的向量

2. **向量对向量求导**  
   若 y 是 m 维向量，x 是 n 维向量，则 ∂y/∂x 是一个 m×n 的Jacobi矩阵

3. **常用公式**  
   - ∂(Ax)/∂x = Aᵀ
   - ∂(xᵀA)/∂x = A
   - ∂(xᵀAx)/∂x = (A + Aᵀ)x
   - ∂(xᵀx)/∂x = 2x

4. **链式法则**  
   对于复合函数 f(g(x))，有：
   ∂f/∂x = (∂f/∂g)(∂g/∂x)

5. **转置规则**  
   (∂y/∂x)ᵀ = ∂yᵀ/∂xᵀ

* 代码实现一元线性回归(以flipper_length为例)
  * 关于改造矩阵
  ```python
  import numpy as np
  flipper_length = np.array(flipper_length)
  y = np.array(body_mass)
  bill_depth = np.array(bill_depth)
  bill_length = np.array(bill_length)
  idx = (bill_length != None) & (bill_depth != None) & (flipper_length != None) & (y != None)
  bill_length = bill_length[idx]
  bill_depth = bill_depth[idx]
  flipper_length = flipper_length[idx]
  y = y[idx]
  bias = np.ones(flipper_length.shape)
  X = np.stack((bias, flipper_length), axis = 1)
  # np.array() 用于将Python列表转换为NumPy数组
  # 参数：需要传入一个Python列表或可迭代对象
  # 例如：flipper_length = [1.2, 3.4, 5.6] -> np.array(flipper_length)

  # np.stack() 用于沿新轴连接数组序列
  # 参数：
  #   - arrays：需要连接的数组序列
  #   - axis：指定连接的轴（0表示垂直堆叠，1表示水平堆叠）
  # 例如：将bias和flipper_length水平堆叠成特征矩阵X

  # np.concatenate() 用于沿现有轴连接数组序列
  # 参数：
  #   - arrays：需要连接的数组序列
  #   - axis：指定连接的轴（0表示垂直连接，1表示水平连接）
  # 与stack的区别：concatenate不会创建新维度，而是沿现有维度连接
  ```
  * 关于手动计算解析解
  ```python
    XTX = np.matmul(X.transpose(), X)
    print("X转置乘X:\n", XTX)
    XTX_inv = np.linalg.inv(XTX)
    w = np.matmul(np.matmul(XTX_inv, X.transpose()), y)
    ```
  * 关于可视化结果
  ```python
    import matplotlib.pyplot as plt
    plt.scatter(flipper_length, y)
    plt.axline(xy1 = (0, w[0]), slope = w[1], color = 'red')
    plt.xlim(np.min(flipper_length), np.max(flipper_length))
    plt.ylim(np.min(y), np.max(y))
    ```
    ![图 6](images/ed5b9c204b629c1d6edffcc964e8c810ced687ae72c05efcea183aa463f01c23.png)  
  * 关于MSE的测度与模型评价L:
    我们先用train测试集的数据得到w，b等相关系数：
    ```python
    权重w: [-5505.90432698    47.4726627 ]
    ```
    我们再导入test测试集（这一步只需要把filename名字做一个修改）
    然后可以得到新的X，y，并且用mse表达公式得到结果：
    ```python
    y_hat = np.matmul(X, w)
    mse = np.mean(np.square(y_hat - y))
    print(mse)
    ```
    这样就能计算出mse。
### 线性回归2.0：多项式回归
* 为什么要多项式回归？
    因为在日常生活中，很少会出现严格的“线性回归”，更多情况下是非线性的。
    而多项式回归本质上只不过是多元回归的一种**特例**，可以理解为多元回归中的每一项“元”都只是x1的一种幂表达。
* 怎么把数学原理做一次推广？
    我们回顾一元线性回归的数学表达式
    $$ y = wx + b $$
    后来我们用矩阵形式改写成了一个矩阵乘法：
    $$ y = X w $$
    这里的X随着样本量堆积成了n行2列的矩阵，一列为1，一列为数据
    w为一个二行一列的列向量，是w和b。
    那我们考虑把这种矩阵形式推广，因为现在的数学形式应该是：
    $$ y =b+ w_1x^1 + w_2x^2 + w_3x^3 +……+ w_nx^n$$
    n那么用矩阵观点来看，现在应该要把矩阵在列数上扩展，因为样本量没变，但是因变量的维度改变了，而w这个列向量则要在行数上扩展，因为是w的每一项代表了系数。
    ![图 7](images/304ba179d2ed694feca2a3a28e70a708363c2b5a5df8e7b69f29915a256c459f.png)  
* 我们代码的细节也因此并不会改变，只需要在初步的数据处理上做一点调整，可视化上做一点调整：
  * 数据处理
  ```python
  import numpy as np

    x1 = np.array(flipper_lengths)
    y = np.array(body_masses)
    x2 = x1 ** 2

    print(x1.shape)

    x0 = np.ones(x1.shape)
    print(x0)
    X = np.stack((x0, x1, x2), axis=1)
    ```
    接下来求解析解的过程没有任何变化，但是注意最后得到的参数w是有顺序的，具有一一对应的关系。
    ```python
    权重w: [ 3.50347651e+04 -3.60558273e+02  1.02243255e+00]
    ```
    这里的mse结果更大了——这里值得思考。
  * 对于可视化，我们或许可以使用更高层级的可视化
  ```python
    # 创建3D图形
    fig = plt.figure()
    # 添加3D子图
    ax = fig.add_subplot(projection='3d')

    # 绘制原始数据点
    # X[:,1]是x1特征，X[:,2]是x2特征，y是目标值
    ax.scatter(X[:,1], X[:,2], y)

    # 创建用于绘制曲面的网格数据
    # rx1和rx2分别在x1和x2的最小最大值之间生成10个等距点
    rx1 = np.linspace(np.min(x1), np.max(x1), 10)
    rx2 = np.linspace(np.min(x2), np.max(x2), 10)
    # 生成网格坐标矩阵
    xx1, xx2 = np.meshgrid(rx1, rx2)
    # 根据回归系数计算预测值
    yy = w[0] + w[1] * xx1 + w[2] * xx2

    # 绘制回归曲面
    # alpha参数控制曲面透明度
    ax.plot_surface(xx1, xx2, yy, alpha=0.1)
    # 设置z轴范围与数据范围一致
    ax.set_zlim(np.min(y), np.max(y))
    ```
    ![图 8](images/30ebe1af9b5b4084f5a60dfb4fc1b165e78c88f5df4139e4b40f625d4e163a55.png)  
    
### 线性回归3.0：多元线性回归
* 对于一个时间，如果有多个因素对其产生影响，我们就需要引入多元变量，这就是
  多元线性回归存在的必要——即我们认为一个事件是由多个因素加权得到的结果，
  这种加权是线性的，而非某种神经网络式的。
* 数学表达：
  $$ y = w_0x_0 + w_1x_1 + w_2x_2 + w_3x_3+……$$
  用矩阵乘法表示即为：
  ![图 9](images/cc8cd3ee77f19cff4217bacf6ea108cdf1ee9f90dd855da9638cf77e8910f9ab.png)  
  可以发现解析解自始至终没有变化
* 代码实现
  * 改进点1：基本数据处理
  ```python
  import numpy as np
  x1 = np.array(flipper_length)
  x2 = np.array(bill_length)
  x3 = np.array(bill_depth)
  y = np.array(body_mass)

  idx = (x2 != None) & (x3 != None) & (x1 != None) & (y != None)
  x1 = x1[idx]
  x2 = x2[idx]
  x3 = x3[idx]
  y = y[idx]
  bias = np.ones(x1.shape)
  x = np.stack((bias, x1, x2, x3), axis = 1)
  ``` 
  * 除此之外的代码几乎没有变化
  * 改进点二：数据预处理——Zscore法和Min-Max法
  我们用多元线性回归得到的系数表达式
  ![图 10](images/36cf115a204fc52d7a46bcf1dbd9b691f7923d5bf06795a73fc047107f870abb.png)  
  但我们能因此就说：$x_3$是最重要的 $x_2$没有$x_1$重要吗？
  我们来看X的数据
  ```python
  [[  1.   39.1  18.7 181. ]
  [  1.   37.8  18.3 174. ]
  [  1.   36.5  18.  182. ]
  [  1.   35.7  16.9 185. ]]
  ```
  其实$x_2$单纯从数据量上将就比$x_1$要大，那么你就不能直接以系数来做比较，一个数据是10000的东西，哪怕他的系数只有0.0几，也不影响这个东西起了决定性作用。所以我们需要把这些数据做*归一化*的处理。
  归一化一般来说有两种思路
    * Min-max法
  $$ X = \frac{X - Xmin}{Xmax - Xmin} $$
    * Z-score法
  $$ X = \frac{X - \mu}{\sigma} $$
    其中$\mu$为均值，$\sigma$为标准差

    * 两种归一化方法的比较：
        1. Min-max法：
            - 将数据线性映射到[0,1]区间
            - 对异常值敏感，容易受极端值影响
            - 适用于数据分布未知或数据范围明确的情况
        2. Z-score法：
            - 将数据转换为均值为0，标准差为1的标准正态分布
            - 对异常值不敏感，适合处理有离群点的数据
            - 适用于数据服从或近似服从正态分布的情况
    * 使用场景：
        - 当数据分布未知或需要保留原始数据比例关系时，使用Min-max法
        - 当数据存在离群点或需要标准化处理时，使用Z-score法
        - 在机器学习中，Z-score法常用于特征缩放，而Min-max法常用于图像处理
  * 代码实现
  ```python
  x1 = (x1 - np.min(x1)) / (np.max(x1) - np.min(x1)) + np.finfo(float).eps
  x2 = (x2 - np.min(x2)) / (np.max(x2) - np.min(x2)) + np.finfo(float).eps
  x3 = (x3 - np.min(x3)) / (np.max(x3) - np.min(x3)) + np.finfo(float).eps 
  ```
  请注意 这里需要加一个np.finfo(float).eps的极小量，预防出现0/0的情况报错
  ```python
  # 计算每个特征的均值和标准差
  x1_mean = np.mean(x1)
  x1_std = np.std(x1)
  x2_mean = np.mean(x2)
  x2_std = np.std(x2)
  x3_mean = np.mean(x3)
  x3_std = np.std(x3)
  x1 = (x1 - x1_mean) / x1_std + np.finfo(float).eps
  x2 = (x2 - x2_mean) / x2_std + np.finfo(float).eps
  x3 = (x3 - x3_mean) / x3_std + np.finfo(float).eps
  ```
  这里记住两个方法：np.mean(), np.std()
  这样处理后的系数就可以代表特征的重要性程度了
  ```python
  [2836.65586543  163.121665   -266.14753242 2100.14533999]
  ```
### 一个改进：关于交叉验证
* 我们现在只有一个测试集和一个训练集
  但是问题在于 这个测试集归根结底只是无数数据中的一批，我们怎么样才能在数据量有限的情况下尽可能泛化呢？
  ![图 11](images/5059268a363b8f2eb66ec48705be5adcd2f93e63fd4f929ced33059ec754b09c.png)  
  我们把$fold_1$作为第一次训练的测试集，剩下四个集作为训练集，得到第一个Accuracy——以此类推，可以得到五个Accuracy.
  通过这样的方式，我们使这个模型更为可靠。
  事实上，我们还可以重复做交叉验证。
  *E.G.*
  我们有三百个样本作为样本集，我们可以先做一次五折交叉验证，然后得到Final_Accuracy，然后可以把这三百个样本打混，然后又得到n组这样的交叉验证。
* 代码实现
  * 如何划分样本集：
  ```python
  idx = np.random.permutation(y.size)
  ##生成一个随机打乱的 0-y.size-1的序列
  mses = []
  for split_idx in range(num_split):
    # make split indice
    tst_idx = set(idx[split_idx::num_split])
    #也就是说 对于上面那个idx中，从idx的第0个开始，
    #每隔num_split记一个，这样就得到了fold1作为测试
    #集元素的index。也就是说得到了一个set，里面
    #装满了目标元素的idx。
    trn_idx = set(idx) - tst_idx
    trn_idx = np.array(list(trn_idx)).astype(int)
    #np生成的数是float
    tst_idx = np.array(list(tst_idx)).astype(int)
    # split data
    X_trn = X[trn_idx]
    #可以直接用np的array作为索引访问X这个更大的nparray
    #顺序访问idx的每个元素，再用这些元素访问X
    y_trn = y[trn_idx]
    X_tst = X[tst_idx]
    y_tst = y[tst_idx]

    X_min = np.min(X_trn, axis=0)
    X_max = np.max(X_trn, axis=0)

    X_trn = (X_trn - X_min)/(X_max - X_min + np.finfo(float).eps)
    x0_trn =  np.ones((X_trn.shape[0], 1))
    X_trn = np.concatenate((x0_trn, X_trn), axis=1)

    result = np.matmul(X_trn.transpose(), X_trn)
    result = np.linalg.inv(result)
    result = np.matmul(result, X_trn.transpose())
    result = np.matmul(result, y_trn)
    w = result

    # test
    
    X_tst = (X_tst - X_min)/(X_max - X_min + np.finfo(float).eps)
    x0_tst =  np.ones((X_tst.shape[0], 1))
    X_tst = np.concatenate((x0_tst, X_tst), axis=1)
    y_tst_pred = np.matmul(X_tst, w)
    
    diff = y_tst - y_tst_pred
    mse = np.mean(diff ** 2)
    mses.append(mse)
    ```
  * 但其实我们发现这么干好像不是很公平，因为你不同的模型生成的随机数——从而生成随机序列——从而选取元素每次都不同
  * 我们希望他们训练的量变大，但基于控制变量的原则，不要两个模型搞出来的比较元素不一样，也就是说，你每次训练的数据应该是同一批。
  ```python
  rng = np.random.default_rng(seed=0)
  #这个seed = 0保证每次生成的随机数序列一致。
  idx = rng.permutation(y.size)
  #相当于你接受了一个指令rng，这个rng.permutation
  #保证调取同一批序列
  ```
### 再改进：梯度下降算法
* 我们看到现在的线性回归模型是解析解的，但是大部分的模型最后恐怕都没有解析解，而且数据维度过高不是什么好事。
* 数学原理：
  在微积分中：
  梯度方向是函数变化的最快方向
  而由于损失函数是下凸的，所以每次往梯度方向减一点，得到新的位置，再往梯度方向减一点……
  直到变化小于某个值（底部应该为0）
  ![图 12](images/35f8cd8e3df6931f35a28628168aee2372e999892129646171bd46920b0b677c.png)  
  * 这个每次下降一点点的系数就是学习率
  如果你的学习率太**高**，你很有可能会导致第一次下降下降过了头，然后这之后因为你的梯度越来越大，映射到了无穷。
  ![图 13](images/278ee293f8e6e86f2e94bc0a5cfe3b44df521bfc6db31aa515c64d5734a2809d.png)  
  如果有点高，则有可能你会错过实际最小值，再一个局部最小值就停手了。
  ![图 14](images/b390b0d6566d9ff2923cd927f773e85ca785e2a57324f983f51c11a2c890b271.png)  
  太小了，耗费时间、算力太久。
  一图以蔽之：
  ![图 15](images/7424158fdb9826545afea9b60cb22175aca19c38336961700e51c534b1e512a8.png)  
  所以你评估的是损失函数，你调整的参数是学习率，目的是为了得到模型最优参数w。
* 代码实现
  ```python
  def calculate_cost(X, y, w):
    d = y - np.matmul(X, w)
    cost = np.mean(d * d)
    return cost
  def calculate_grad(X, y, w):
    grad = np.matmul(np.matmul(X.transpose(), X), w)
    grad = grad - np.matmul(X.transpose(), y)
    grad = grad * 2 / X.shape[0]
    return grad
  ```
  这里是实现了前馈和梯度计算
  ```python
  mu, sigma = 0., 1.
  w_t = np.random.normal(mu, sigma, size=4).reshape((4, 1))
  print(w_t)
  ```
  高斯分布生成第一个w用于第一次下降。
  ```python
  Js = []

  alpha = 0.1##learning_rate
  for t in range(5000):#学习上限
    g_t = calculate_grad(X, y, w_t)
    w_t -= alpha * g_t

    J_t = calculate_cost(X, y, w_t)
    Js.append(J_t)

  print(w_t)
  ```


* 但是，实际操作中我们发现X的样本太多时候向量乘积是不可能的，所以如果我们只取一个点，那么就形成了**SGD**
  ![图 0](images/e9af6dd2468b29dd8f9b7106b822ec472b6bd99c7faac1191c08e13d437804a6.png)  
也就是说损失函数只按一个样本下降，不考虑全部的loss函数。
但很快发现这样很有可能被某一两个噪声值影响过大，从而导致拟合精度不高、整体上的损失过大。
所以我们考虑折中，引入一个batch来做小规模样本的梯度下降
![图 1](images/e0525bd5a8617a2864b4f064511b6fb665fdee17a405e719ac02c89485976f03.png)  
* 代码实现
  ```python
  batch_size=10

  for alpha in (0.1, 0.01):
    for epoch in range(500):
        idx = np.random.permutation(X.shape[0])
        for row_idx in idx[::batch_size]:
            x_t = X[row_idx:row_idx+batch_size]
            y_t = y[row_idx:row_idx+batch_size]
  ```
  也即 每次只下降一小部分 但是最后总归是下降完了所有的可能性。
  
