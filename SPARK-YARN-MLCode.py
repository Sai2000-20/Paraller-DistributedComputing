from operator import add
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark import SparkFiles

spark = SparkSession.builder \
    .master("spark://172.31.92.167:7077") \
    .appName("Saira Testing") \
    .getOrCreate()

df = spark.read.csv("file://"+SparkFiles.get('/home/ubuntu/Cleaned-Data.csv'), header=True, inferSchema= True)

df.printSchema()


# In[36]:


#drop country
df.printSchema()
df = df.drop('Country')
df.printSchema()


# In[37]:


from pyspark.sql.functions import isnull, when, count, col

#missing values check
#df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).show()


# In[38]:


#df.groupby('Fever').count().toPandas()


# In[39]:


#df.describe().show()


# In[40]:


from pyspark.sql.functions import *

newDf = df.withColumn('Severity_Mild', regexp_replace('Severity_Mild', '1', '2')) .withColumn('Severity_Moderate', regexp_replace('Severity_Moderate', '1', '3')) .withColumn('Severity_Severe', regexp_replace('Severity_Severe', '1', '4')) .withColumn('Age_0-9', regexp_replace('Age_0-9', '1', '1')) .withColumn('Age_10-19', regexp_replace('Age_10-19', '1', '2')) .withColumn('Age_20-24', regexp_replace('Age_20-24', '1', '3')) .withColumn('Age_25-59', regexp_replace('Age_25-59', '1', '4')) .withColumn('Age_60+', regexp_replace('Age_60+', '1', '5')) .withColumn('Contact_Dont-Know', regexp_replace('Contact_Dont-Know', '1', '2')) .withColumn('Contact_No', regexp_replace('Contact_No', '1', '1')) .withColumn('Contact_Yes', regexp_replace('Contact_Yes', '1', '3'))

newDf.printSchema()


# In[41]:


from pyspark.sql.types import IntegerType

newDf = newDf.withColumn("Severity_Mild", newDf["Severity_Mild"].cast(IntegerType())) .withColumn("Severity_Moderate", newDf["Severity_Moderate"].cast(IntegerType())) .withColumn("Severity_Severe", newDf["Severity_Severe"].cast(IntegerType())) .withColumn("Age_0-9", newDf["Age_0-9"].cast(IntegerType())) .withColumn("Age_10-19", newDf["Age_10-19"].cast(IntegerType())) .withColumn("Age_20-24", newDf["Age_20-24"].cast(IntegerType())) .withColumn("Age_25-59", newDf["Age_25-59"].cast(IntegerType())) .withColumn("Age_60+", newDf["Age_60+"].cast(IntegerType())) .withColumn("Contact_Dont-Know", newDf["Contact_Dont-Know"].cast(IntegerType())) .withColumn("Contact_No", newDf["Contact_No"].cast(IntegerType())) .withColumn("Contact_Yes", newDf["Contact_Yes"].cast(IntegerType()))


# In[42]:


newDf = newDf.drop('None_Sympton') .drop('None_Experiencing')
newDf.printSchema()


# In[43]:


newDf = newDf.withColumn("Score", col("Fever")+col("Tiredness")+col("Dry-Cough")+col("Difficulty-in-Breathing")+
                         col("Sore-Throat")+col("Pains")+col("Nasal-Congestion")+col("Runny-Nose")+col("Diarrhea")+
                         col("Age_0-9")+col("Age_10-19")+col("Age_20-24")+col("Age_25-59")+col("Age_60+")+
                         col("Gender_Female")+col("Gender_Male")+col("Gender_Transgender")+col("Severity_Mild")+
                         col("Severity_Moderate")+col("Severity_None")+col("Severity_Severe")+col("Contact_Dont-Know")+
                         col("Contact_No")+col("Contact_Yes")) 


# In[44]:


newDf.toPandas()


# In[45]:


from pyspark.ml.feature import Bucketizer
bucketizer = Bucketizer(splits=[ 0,6,11,16, float('Inf') ],inputCol="Score", outputCol="bucket")
df_buck = bucketizer.setHandleInvalid("keep").transform(newDf)

df_buck.toPandas()


# In[46]:


df_buck.select([count(when(isnull(c), c)).alias(c) for c in df_buck.columns]).show()


# In[47]:


df_buck.groupby('bucket').count().toPandas()


# In[48]:


from pyspark.sql.functions import array, col, lit
labels = ["None", "Mild","Moderate", "Severe"]
label_array = array(*(lit(label) for label in labels))

new = df_buck.withColumn("Conditions", label_array.getItem(col("bucket").cast("integer")))
new.toPandas()


# In[49]:


new.select([count(when(isnull(c), c)).alias(c) for c in new.columns]).show()


# In[50]:


newDf = new.drop('bucket') .drop('Score')

newDf.printSchema()


# In[51]:


#newDf.toPandas().to_csv('mycsv.csv')


# In[52]:


# Split the data
#(training_data, test_data) = df.randomSplit([0.7,0.3], seed =2020)
#print("Training Dataset Count: " + str(training_data.count()))
#print("Test Dataset Count: " + str(test_data.count()))


# In[53]:


new.select([count(when(isnull(c), c)).alias(c) for c in new.columns]).show()


# In[54]:


from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

# transformer
vector_assembler = VectorAssembler(inputCols=["Fever", "Tiredness", "Dry-Cough", 
                                              "Difficulty-in-Breathing", "Sore-Throat", "Pains", 
                                              "Nasal-Congestion", "Runny-Nose", "Diarrhea", "Gender_Female", 
                                              "Gender_Male", "Gender_Transgender", "Severity_Severe",
                                              "Severity_Mild", "Severity_Moderate","Severity_None",
                                              "Age_0-9","Age_10-19","Age_20-24","Age_25-59","Age_60+",
                                              "Contact_Dont-Know","Contact_No","Contact_Yes"],outputCol="features")
df_temp = vector_assembler.transform(newDf)
df_temp.show(3)

# drop the original data features column
df = df_temp.drop("Fever", "Tiredness", "Dry-Cough", "Difficulty-in-Breathing", "Sore-Throat", "Pains", 
                  "Nasal-Congestion", "Runny-Nose", "Diarrhea", "Gender_Female", "Gender_Male", "Gender_Transgender", 
                  "Severity_Severe","Severity_Mild", "Severity_Moderate","Severity_None",
                  "Age_0-9","Age_10-19","Age_20-24","Age_25-59","Age_60+",
                  "Contact_Dont-Know","Contact_No","Contact_Yes")
df.show(5)


# In[55]:


df.printSchema()


# In[56]:


from pyspark.ml.feature import StringIndexer

# estimator
#l_indexer = StringIndexer(inputCol="Conditions", outputCol="labelIndex")
#df = l_indexer.fit(df).transform(df)
#df.show(10)

indexer = StringIndexer(inputCol="Conditions", outputCol="labelIndex")
df = indexer.fit(df).transform(df)
df.show(10)
print("\n\nThe label index for each class are : 0 = Moderate , 1 = Mild , 2 = Severe , 3 = None")


# In[57]:


df.select([count(when(isnull(c), c)).alias(c) for c in df.columns]).show()


# In[58]:


df.toPandas().to_csv('new.csv')


# In[59]:


# data splitting
(training,testing)=df.randomSplit([0.7,0.3])
print("Training Dataset Count: " + str(training.count()))
print("Test Dataset Count: " + str(testing.count()))


# ## Random Forest

# In[60]:


# accuracy score
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# train our model using training data
rf = RandomForestClassifier(labelCol="labelIndex",featuresCol="features", numTrees=10)
model = rf.fit(training)

# test our model and make predictions using testing data
predictions = model.transform(testing)
predictions.select("prediction", "labelIndex").show(5)

# evaluate the performance of the classifier
evaluator = MulticlassClassificationEvaluator(labelCol="labelIndex",predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print("RF Test Error = %g" % (1.0 - accuracy))
print("RF Accuracy = %g " % accuracy)


# In[61]:


# confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

y_true = predictions.select(['labelIndex']).collect()
y_pred = predictions.select(['prediction']).collect()

print(classification_report(y_true, y_pred, target_names=['None', 'Mild', 'Moderate', 'Severe']))


# ## Decision Tree

# In[62]:


# accuracy score
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# train our model using training data
dt = DecisionTreeClassifier(labelCol="labelIndex", featuresCol="features")
model = dt.fit(training)

# test our model and make predictions using testing data
predictions = model.transform(testing)
predictions.select("prediction", "labelIndex").show(5)

# evaluate the performance of the classifier
evaluator = MulticlassClassificationEvaluator(labelCol="labelIndex",predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print("DT Test Error = %g" % (1.0 - accuracy))
print("DT Accuracy = %g " % accuracy)


# In[63]:


# confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

y_true = predictions.select(['labelIndex']).collect()
y_pred = predictions.select(['prediction']).collect()

print(classification_report(y_true, y_pred, target_names=['None', 'Mild', 'Moderate', 'Severe']))


# ## Naive Bayes

# In[64]:


from pyspark.ml.classification import NaiveBayes

# train our model using training data
nb = NaiveBayes(labelCol="labelIndex", featuresCol="features")
model = nb.fit(training)

# test our model and make predictions using testing data
predictions = model.transform(testing)
predictions.select("prediction", "labelIndex").show(5)

# evaluate the performance of the classifier
evaluator = MulticlassClassificationEvaluator(labelCol="labelIndex",predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print("NB Test Error = %g" % (1.0 - accuracy))
print("NB Accuracy = %g " % accuracy)


# In[65]:


# confusion matrix

y_true = predictions.select(['labelIndex']).collect()
y_pred = predictions.select(['prediction']).collect()

print(classification_report(y_true, y_pred, target_names=['None', 'Mild', 'Moderate', 'Severe']))

