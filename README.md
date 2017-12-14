# sklearn1
<ul>
<li>python 3.5.2</li>
<li>numpy 1.11.0</li>
<li>matplotlib 1.5.1</li>
<li>pandas 0.17.1</li>
<li>scipy 0.17.0</li>
<li>scikit-learn 0.19.1</li>
</ul>
### Perceptron
<div style="background:#333;">
python3 sklearn1.py
<hr>
使用sklearn集成的Perceptron和iris数据集训练一个感知器
</div>
<br>
<div style="background:#333;">
python3 drawline.py
<hr>
绘制一段sigmoid曲线
</div>
<br>
<div style="background:#333">
python3 sklearn2_logistic.py
<hr>
使用sklearn的logistic regression训练一个模型，该模型的损失函数引入了L2正则项以降低过拟合的风险。
</div>

第一，引入了条件概率，从预测标签变成了预测概率；
<br>
第二，激活函数变成了logistic；
<br>
第三，激活函数和logit(p)互为反函数，很好推导；

$$\phi(z)=p=\frac{1}{1+e^{-z}}$$
<br>

$$logit(p(y=1|x))=z=log \frac{p}{1-p}$$

第四，模型改变导致损失函数改变，首先构造出单个样本的似然函数，再构造样本集的对数似然，最优预测就是最大似然，所以把似然函数变成对数似然方便求最大值。求最大值和求最小值没有区别，且梯度下降适合求极小值，所以损失函数取负对数似然。
<br>
似然函数

$$L(\bold w)=P(\bold y|\bold x;\bold w)=\Pi_{i=1}^nP(y^{(i)}|x^{(i)};\bold w)=(\phi(z^{(i)}))^{y^{(i)}}(1-\phi(z^{(i)}))^{1-y^{(i)}}$$
<br>
对数似然

$$l(\bold w)=logL(\bold w)=\sum_{i=1}^n y^{(i)}log(\phi(z^{(i)}))+(1-y^{(i)})log(1-\phi(z^{(i)}))$$
<br>
损失函数

$$J(\bold w)=\sum_{i=1}^n -y^{(i)}log(\phi(z^{(i)}))-(1-y^{(i)})log(1-\phi(z^{(i)}))$$
<br>

以单个样本为例，来说明该损失函数确实是在模型最优时收敛

$$J(\phi(z),y;\bold w)=\begin{cases}
-log(\phi(z))\ \ \ \ \ if\ y=1\\
-log(1-\phi(z))\ \ \ if\ y=0
\end{cases}$$
$$\phi(z)\in(0,1]$$
$$if\ \ y=1,\ -log(\phi(z))\in(0,\infty )=\begin{cases}
0,if\ \phi(z)=1,true\\
\infty,if\ \phi(z)=0,false
\end{cases}$$
$$if\ \ y=0,\ -log(1-\phi(z))\in(0,\infty)=\begin{cases}
\infty,if\ \phi(z)=1,false\\
0,if\ \phi(z)=0,true
\end{cases}$$
可以看到预测正确时，损失函数不增加，预测错误时损失函数增加，所以模型（趋于）最优时，损失函数收敛。
<br>
## L2 regularization

在损失函数中加入L2正则项，来降低过拟合

L2正则项

$$\frac{\lambda}{2}||\bold w||^2=\frac{\lambda}{2}\sum_{j=1}^m w_j^2$$

正则化系数

$$\lambda$$
<br>
损失函数

$$C=\frac{1}{\lambda}$$

$$J(\bold w)=C\{-\sum_{i=1}^n[y^{(i)}log(\phi(z^{(i)})+(1-y^{(i)})(1-\phi(z^{(i)}))]\}+\frac{1}{2}||\bold w||^2$$

从mse那篇的推导中可以看出误差的构成包括　偏差bias，方差var，系统误差eta，这三个部分。
在损失函数中加入L2正则项，python machine learning书中称该项为bias，由于形式上和mse推导出来的多项式差异较大，以后再去深究是什么项。但是从mse的推导中，我们已经看到三个部分对于模型的影响，并且知道了bias与模型复杂度成反比，var与模型复杂度成正比。
一个较为直观的理解是：L2正则项的加入，在权重更新时，权重值相比原来，减少的更多，这会使某些导致over fitting的权重减少更多（虽然有用的权也被减少了许多）。我们知道权重是导致超平面波动的内因，大的权重会导致波动剧烈，所以在惩罚权重时，我们不在乎小的权重（因为足够小就意味着足够贴近理想的超平面，也就越稳定），而更关注导致波动剧烈（over fitting）的大权重，所以L2的加入虽有一定影响，但是惩罚那些大权重的作用更突出。
