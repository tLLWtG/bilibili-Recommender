# bilibili-Recommender

一个基于 bilibili 用户行为数据和深度学习的个性化视频推荐系统。

> 声明：bilibili-Recommender 仅用于学习和测试

本项目调用 bilibili 网页 API 获取登录用户的观看、收藏、投币、点赞等行为数据，并利用这些数据实时训练个性化推荐模型，深入分析用户的兴趣画像。

在功能上，我们实现了从热门视频榜单中，利用训练的模型个性化精选出用户感兴趣的视频。另外也可以推荐非热门视频，帮助用户发现更多可能感兴趣的潜在内容。推荐模型推荐的视频都会同时给出推荐指数（Rating）以供参考。

## 1. Developer Team

* [tLLWtG](https://github.com/tLLWtG)（编写后端框架、设计前端页面）
* [wegret](https://github.com/wegret)（构建推荐模型、部分后端代码）
* [corgiInequation](https://github.com/corgiInequation)（对接 bilibili 网站的相关 API、部分后端代码）

## 2. 环境配置及运行

> 项目使用的 python 版本为 3.9.20，用到的库见 `requirements.txt`，可按照下面的步骤自动配置。

1. 在虚拟 python 环境中安装 requirements.txt 里的库：`pip install -r requirements.txt`。
2. 在命令行中运行 bilibili-Recommender 的启动脚本：`python ./run.py`。
3. 点击命令行中显示的链接，打开前端页面。

## 3. 技术细节

### 3.1 用户行为数据获取

本项目调用 bilibili 网页 API 获取登录用户的观看、收藏、投币、点赞等行为数据（使用的 bilibili API 来源于 [bilibili-API-collect](https://github.com/SocialSisterYi/bilibili-API-collect)）。之后进行简单的预处理，并整理成Json文件，方便推荐模型处理。

相较于常见用户手动打分的方案，系统自动获取用户行为数据可以提供更加流畅、无缝的用户体验，也为之后的基于深度学习的推荐模型提供了更多维度的特征（点赞、收藏、观看时长等）。

### 3.2 前后端

Flask+Jinja2 的方案，这里简要介绍以下几个核心内容。

* **前后端数据交互**
  
  通过 Flask 封装的路由功能，我们将 URL 映射到具体的 python 函数，前端要获取数据或是发送数据时只需要访问对应的URL（例如获取推荐视频列表、发送Logout信号）。

* **模拟登陆操作**
  
  后端调用 bilibili 网页登陆 api，然后将相关信息合成可以识别的登陆码存于二维码中，再将二维码图片传给前端。这样用手机 app 扫描后，后端即可模拟登陆操作，获取用户的 cookie。

* **获取用户行为数据**
  
  利用登陆时获取的用户 cookie，然后使用 requests 库访问对应的 api 即可获取用户行为数据（历史观看、收藏等）。之后将其整理成方便展示和适合推荐模型处理的 json 格式。

### 3.3 推荐模型

考虑到只能获取当前用户的数据，因此使用基于物品的推荐方法。除此之外，希望模型能支持支持在线学习。因此模型方案选用：**Deep & Wide + Attention**（基于 Tensorflow 库）。

#### 特征工程

对于模型需要提取的特征，我们划分了以下三类：

1. 内容类型特征：标签向量和作者特征，这里表示了这个样本主要吸引的人群范围。对于作者特征，这里将作者视为类型变量。标签向量的处理，这里考虑标签特征通过embedding层处理，使用attention机制关注重要的标签，作者信息同理。因为要考虑在线学习机制，所以这里embedding层要求能动态重建。

2. 内容质量特征：总浏览量、点赞数、收藏数，这里可能要做一个比例，计算出来内容质量评分。

   $$\text{Score} = \alpha \times \left( \frac{\text{Likes}}{\log{(\text{Views +1})}} \right)+ \beta \times \left( \frac{\text{Favs}}{\log{(\text{Views +1})}} \right) + \gamma \times \log(\text{Views}+1)$$

3. 用户兴趣画像：对于每个视频，当前用户的观看进度，是否点赞，是否收藏，从而计算出来用户的感兴趣评分。
   
   $$\text{Interest} = \max \left( \text{isliked} + \text{isfaved}, \frac{\text{progress}}{\text{duration}}\right)$$

#### 模型构建

提取特征后，经过 deep 部分和 wide 部分。

* **deep 部分**
  
  主要处理内容类型（或者说视频面向的兴趣范围）特征。输入层中，标签通过可扩展的Embedding层，转为低维稠密向量，作者ID也是，转成了作者向量，然后两个embedding拼接在一起做一个组合特征`combined_embedded`。而质量分数直接作为数值特征输入。Attention层，使用注意力机制来处理标签序列，前面的组合特征`combined_embedded`会被输入进来，计算不同的标签的重要性权重，然后得到一个综合的内容特征向量。Deep layer层，包含了3个全连接层。
* **wide 部分**
  
  主要是处理内容质量特征，通过一个简单的线性层直接学习内容质量分数和用户兴趣之间的关系，用来尝试捕捉一些比较明显的特征关系。线性变化公式为：`wide_output = quality_score * w + b`

最后将Wide部分和Deep部分的输出连接起来，通过一个sigmoid激活函数的全连接层得到最终的预测分数。

另外考虑到正负样本可能存在不平衡问题（事实上确实存在存在此问题，因为用户点赞信息相比总共的浏览记录是比较少的），在实际训练时，还加上了类别权重：`class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)`

## 4. License

This project is licensed under the MIT license. External libraries used by bilibili-Recommender are licensed under their own licenses.
