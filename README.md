# Social Media Sentiment Analysis with Apache Spark

## Problem

In this repo we will be building a #Social #Media #Sentiment #Analysis application with Apache #Spark. We will not be focusing on the #NLP and #Modeling part for now. We will be doing data preprocessing using Scala and Apache Spark, and we will be classifying tweets as positive or negative using a Gradient Boosting algorithm. Although this is focused on sentiment analysis, Gradient Boosting is a versatile technique that can be applied to many classification problems. You should be able to reuse this code to classify text in many other ways, such as spam or not spam, news or not news, ...etc.

## Data

You can find it at src/test/resources/data/tweets.json

## Run 

cd into the project. You get the following files after doing so.

```linux 
build.sbt  project  README.md  run.sh  spark-warehouse  src  target
```

Tape the following command to package the project as a JAR file

```linux
$ sbt package
```

Run the resultant package with spark-submit

* Method 1: If you cannot reach spark-submit from the current directory

```linux
$ path/to/spark/bin/spark-submit --class "SAApp" --master local[2] target/scala-2.11/saapp_2.11-0.1.jar
```

* Method 2: Otherwise

```
$ ./run.sh
```

## End note 

Hope this can help! The techniques demonstrated can be directly applied to other text classification models, such as spam classification. Try running this code with other keywords besides happy and sad and see what models you can build.
