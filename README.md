# SVHN-mat-tfrecord

### 项目介绍

实现了一个可以把svhn数据集从.mat类型的文件直接转换成.tfrecord格式的文件.(个人认为后续写导入接口会比较方便)

### 关于svhn数据集:

> SVHN is a real-world image dataset for developing machine learning and object recognition algorithms with minimal requirement on data preprocessing and formatting. It can be seen as similar in flavor to MNIST (e.g., the images are of small cropped digits), but incorporates an order of magnitude more labeled data (over 600,000 digit images) and comes from a significantly harder, unsolved, real world problem (recognizing digits and numbers in natural scene images). SVHN is obtained from house numbers in Google Street View images. <br>

And here we use the second format.<br>

![SVHN](http://ufldl.stanford.edu/housenumbers/32x32eg.png)<br>

更多详细的信息可以到[svhn数据集](http://ufldl.stanford.edu/housenumbers/).