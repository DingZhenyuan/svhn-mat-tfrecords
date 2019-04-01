# SVHN-mat-tfrecord

> SVHN is a real-world image dataset for developing machine learning and object recognition algorithms with minimal requirement on data preprocessing and formatting. It can be seen as similar in flavor to MNIST (e.g., the images are of small cropped digits), but incorporates an order of magnitude more labeled data (over 600,000 digit images) and comes from a significantly harder, unsolved, real world problem (recognizing digits and numbers in natural scene images). SVHN is obtained from house numbers in Google Street View images. <br>

And here we use the second format.<br>

![SVHN](http://ufldl.stanford.edu/housenumbers/32x32eg.png)<br>

What I have done is completing a method to convert the svhn dataset from the .mat format to the .tfrecords format.<br>
Don't you think it is convenient? You can just go to use `TFRecordWriter()` to generate a reader!