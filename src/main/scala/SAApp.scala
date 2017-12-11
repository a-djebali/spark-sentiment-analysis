/**
  * STACK: Libraries and functionalities we need along the way  
  */

import org.apache.spark._
import org.apache.spark.rdd._
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.sql.SparkSession
import scala.util.{Success, Try}

object SAApp {

  // ==> SPARK SESSION: Creating a SparkSession automatically gets SparkContext as part of it.
  // At this point we can use the sparkSession variable as an instance object to access 
  // its public methods and instances for the duration of Spark job.

  val sparkSession = SparkSession
    .builder()
    .appName("Spark Sentiment Analysis Application")
    .getOrCreate()

  def main(args: Array[String]) {

    /**
      * LOAD DATA INTO SPARK: Here data is loaded from HDFS, but you can find it at src/test/resources/data/tweets.json
      */

    val tweetDF = sparkSession.sqlContext
      .read
      .json("hdfs://localhost:54310/tmp/data_staging/*") // Or load it as follows ("../../test/resources/data/*")
    
    // ==> HINT: the following command loads back separated .json files into one merged file in a local dir 
    // linux$ hadoop dfs -getmerge /tmp/data_staging/* /path/to/local/dir/tweets.json

    /* Sample of the data 

    scala> tweetDF.show(4)

    Output:

    +---------------+--------------------+----------------+---------------+----+--------------------+--------------------+------------------+
    |_corrupt_record|        created_time|created_unixtime|    displayname|lang|                 msg|           time_zone|          tweet_id|
    +---------------+--------------------+----------------+---------------+----+--------------------+--------------------+------------------+
    |           null|Wed Mar 15 00:32:...|   1489537963226|  meantforpeace|  en|ogmorgaan xo_rare...|Eastern Time (US ...|841809347768377344|
    |           null|Wed Mar 15 00:32:...|   1489537963229|        mushtxq|  en|Happy early birth...|              London|841809347780972544|
    |           null|Wed Mar 15 00:32:...|   1489537962590| Carla_Pereira2|  en|After 10 hrs on a...|                    |841809345100697600|
    |           null|Wed Mar 15 00:32:...|   1489537963311|     DillMaddie|  en|i just want to be...|                    |841809348124917761|
    +---------------+--------------------+----------------+---------------+----+--------------------+--------------------+------------------+
    */

    /**
      * DATA PREPROCESSING: Clean, label and transform tweets to be ready for modeling 
      */

    // ==> DATA CLEANING: To have clean data, we want to remove tweets that don't contain "happy" 
    // or "sad" and select an equal number of happy and sad tweets to prevent bias in the model later on. 

    // get all messages
    var messages = tweetDF.select("msg")

    // remove tweets that don't contain "happy" word
    var happyMessages = messages.filter(messages("msg").contains("happy"))
    val countHappy = happyMessages.count()

    // remove tweets that don't contain "sad" word
    var unhappyMessages = messages.filter(messages("msg").contains(" sad"))
    val countUnhappy = unhappyMessages.count()

    // number of messages to consider
    val smallest = Math.min(countHappy, countUnhappy).toInt

    // create a dataset with equal parts happy and unhappy messages
    var tweets = happyMessages.limit(smallest).unionAll(unhappyMessages.limit(smallest))

    // ==> DATA LABELING: Here we label each happy tweet as 1 and unhappy ones as 0. We also split 
    // each tweet into a collection of words.

    // convert the Dataframe to an RDD to easily transform data using the map function.
    val messagesRDD = tweets.rdd

    // filter out tweets that cannot be parsed
    val goodBadRecords = messagesRDD.map(
      row =>{
        Try{
          val msg = row(0).toString.toLowerCase()
          var isHappy:Int = 0
          if(msg.contains(" sad")){
            isHappy = 0
          }else if(msg.contains("happy")){
            isHappy = 1
          }
          var msgSanitized = msg.replaceAll("happy", "")
          msgSanitized = msgSanitized.replaceAll("sad","")

          // return tuples tuples of the form (Int, Seq[String]), where a 1 for the first term indicates happy 
          // and 0 indicates sad. The second term is a sequence of words and emojis.
          (isHappy, msgSanitized.split(" ").toSeq)
        }
      }
    )

    // filter out labeled tweets from wrong ones (exceptions)
    var labeledTweets = goodBadRecords.filter((_.isSuccess)).map(_.get)
    
    // ==> DATA TRANSFORMATION: A machine learning model expects a vector as an input (feature array) 
    // To do so we use the hashing trick, in which we hash each word and index it into a fixed-length array.

    val hashingTF = new HashingTF(2000)

    // map the input strings to a tuple of labeled point + input text
    val input_labeled = labeledTweets.map(
      t => (t._1, hashingTF.transform(t._2)))
      .map(x => new LabeledPoint((x._1).toDouble, x._2))

    /* Sample of the data after preprocessing 

    scala> input_labeled.take(3).foreach(println)

    Output:

    (1.0,(2000,[405,775,1372,1754,1993],[1.0,1.0,1.0,1.0,1.0]))
    (1.0,(2000,[163,201,343,433,580,584,630,656,899,1138,1310,1320,1330,1371,1372,1400,1425,1433,1541,1543,1643,1746,1760],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]))
    (1.0,(2000,[44,163,170,213,237,388,430,656,674,688,781,1036,1076,1329,1368,1371,1372,1425,1526,1564,1604,1760],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,4.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0,1.0,2.0,1.0,1.0,1.0,1.0,1.0]))
    */

    /**
      * MODELING: Here when science comes along
      */  

    // ==> OVERFITTING: Split the data into training and testing set in a way to have a predictive model 
    // that will generalized well (high accuracy on the training set as well as the testing set)

    // Split the data as 70% for training and 30% as a testing set
    val splits = input_labeled.randomSplit(Array(0.7, 0.3))
    val (trainingData, testingData) = (splits(0), splits(1))
    
    // training the model 
    val boostingStrategy = BoostingStrategy.defaultParams("Classification")

    // number of passes over our training data
    boostingStrategy.setNumIterations(20) 

    // we have two output classes: happy and sad
    boostingStrategy.treeStrategy.setNumClasses(2) 

    // depth of each tree. Higher numbers mean more parameters, which can cause overfitting.
    // lower numbers create a simpler model, which can be more accurate.
    // in practice we have to tweak this number to find the best value.
    boostingStrategy.treeStrategy.setMaxDepth(5)

    val model = GradientBoostedTrees.train(trainingData, boostingStrategy)

    /**
      * PREDICTIONS
      */
 
    // get a sample of the data in its native format after labeling it
    var sample = labeledTweets.take(1000).map(
      t => (t._1, hashingTF.transform(t._2), t._2))
      .map(x => (new LabeledPoint((x._1).toDouble, x._2), x._3))

    val predictions = sample.map { point =>
      val prediction = model.predict(point._1.features)
      (point._1.label, prediction, point._2)
    }

    //The 1st entry is the true label. 1 is happy, 0 is unhappy. The 2nd entry is the prediction.
    predictions.take(100).foreach(x => println("label: " + x._1 + " prediction: " + x._2 + " text: " + x._3.mkString(" ")))

    /* Sample of the predictions 
    
    label: 1.0 prediction: 0.0 text: rt mcasalan pinatawag pala cyamarydaleentrat5 many more blessings to come baby girlsuper  talaga kami para sainyo ni ed…
    label: 1.0 prediction: 1.0 text: rt dmtroll without this guy we wont have little  moments in the latest episodes of defendant
    label: 1.0 prediction: 1.0 text: i  birthday my guy dj_d_rac the bushes seem very comfortable to sleep in lolol
    label: 1.0 prediction: 1.0 text: rt maywarddvo_thai trust yourself create the kind of self that you will be  to live with all your lifembmayward
    label: 1.0 prediction: 1.0 text: rt tyler_labedz dont forget the  thoughts
    label: 1.0 prediction: 0.0 text: rt kia4realz im legit  for cardi bs success bxfinest
    label: 0.0 prediction: 1.0 text:  birthday gronk  no one ever wished you a  birthday but youre the reason i started watching footb…
    label: 1.0 prediction: 0.0 text: u dont have to say tht u miss me at all i just wanted to let u know tht i do miss u n it sucks to see u  n im not the reason why      
    
    */

    /**
      * EXPORT THE MODEL: Cuz the model can be used for production and server business applications  
      */ 

    // Once the model is as accurate, it can be exported for production use. 
    // e.g. can be easily loaded back into a Spark Streaming workflow for use in production.
    // model.save("hdfs://localhost:54310/tmp/models/SentimentAnalysisGradientBoosting")

    // stop the sparkSession
    sparkSession.stop()
  }
}
