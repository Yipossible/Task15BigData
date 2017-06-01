
# coding: utf-8

# In[4]:

import findspark
findspark.init()
import pyspark
import random
sc = pyspark.SparkContext(appName="Pi")
num_samples = 100000000
def inside(p):     
  x, y = random.random(), random.random()
  return x*x + y*y < 1
count = sc.parallelize(range(0, num_samples)).filter(inside).count()
pi = 4 * count / num_samples
print(pi)
sc.stop()


# In[5]:

import findspark
findspark.init()

from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession

from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType

spark = SparkSession.builder.appName("test").getOrCreate()

sentenceDataFrame = spark.createDataFrame([
    (0, "Hi I heard about Spark"),
    (1, "I wish Java could use case classes"),
    (2, "Logistic,regression,models,are,neat")
], ["id", "sentence"])

tokenizer = Tokenizer(inputCol="sentence", outputCol="words")

regexTokenizer = RegexTokenizer(inputCol="sentence", outputCol="words", pattern="\\W")
# alternatively, pattern="\\w+", gaps(False)

countTokens = udf(lambda words: len(words), IntegerType())


tokenized = tokenizer.transform(sentenceDataFrame)
tokenized.select("sentence", "words")    .withColumn("tokens", countTokens(col("words"))).show(truncate=False)

regexTokenized = regexTokenizer.transform(sentenceDataFrame)
regexTokenized.select("sentence", "words")     .withColumn("tokens", countTokens(col("words"))).show(truncate=False)


# In[16]:

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, Tokenizer

# Prepare training documents from a list of (id, text, label) tuples.
training = spark.createDataFrame([
    (0, "a b c d e spark", 1.0),
    (1, "b d", 0.0),
    (2, "spark f g h", 1.0),
    (3, "hadoop mapreduce", 0.0)
], ["id", "text", "label"])

# Configure an ML pipeline, which consists of one stages: tokenizer, hashingTF, and lr.
tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol="features")
lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
pipeline = Pipeline(stages=[tokenizer, hashingTF, lr])

# Fit the pipeline to training documents.
model = pipeline.fit(training)

# Prepare test documents, which are unlabeled (id, text) tuples.11111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111
test = spark.createDataFrame([
    (4, "spark i j k"),
    (5, "l m n"),
    (6, "spark hadoop spark"),
    (7, "apache hadoop")
], ["id", "text"])

# Make predictions on test documents and print columns of interest.
prediction = model.transform(test)
selected = prediction.select("id", "text", "probability", "prediction")
for row in selected.collect():
    rid, text, prob, prediction = row
    print("(%d, %s) --> prob=%s, prediction=%f" % (rid, text, str(prob), prediction))


# In[14]:

import findspark
findspark.init()

from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession

df = spark.read.load("/Users/yiwang/Documents/YiWang/Ebiz/Task 15/attributes.csv", format="csv")

df.show()


# In[83]:

import findspark
findspark.init()

from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.types import StructType, StructField
from pyspark.sql.types import DoubleType,StringType,IntegerType

sc = spark.sparkContext
sql_sc = SQLContext(sc)

trainSchema = StructType([
    StructField("id", IntegerType()),
    StructField("pid", IntegerType()),
    StructField("title", StringType()),
    StructField("term", StringType()),
    StructField("score", DoubleType())
])

titleSchema = StructType([
    StructField("pid", IntegerType()),
    StructField("title", StringType())
])

descriptionSchema = StructType([
    StructField("pid", IntegerType()),
    StructField("description", StringType())
])

attrSchema = StructType([
    StructField("pid", IntegerType()),
    StructField("name", StringType()),
    StructField("value", StringType()),
])

title = sql_sc.read.format("com.databricks.spark.csv").option("header","true").schema(trainSchema).load("/Users/yiwang/Documents/YiWang/Ebiz/Task 15/train.csv")
train = sql_sc.read.format("com.databricks.spark.csv").option("header","true").schema(titleSchema).load("/Users/yiwang/Documents/YiWang/Ebiz/Task 15/RawTrain.csv")
test=sql_sc.read.format("com.databricks.spark.csv").option("header","true")    .schema(dataSchema).load("/Users/yiwang/Documents/YiWang/Ebiz/Task 15/test.csv")
attr = sql_sc.read.format("com.databricks.spark.csv").option("header","true").schema(attrSchema).load("/Users/yiwang/Documents/YiWang/Ebiz/Task 15/attributes.csv")

description = sql_sc.read.format("com.databricks.spark.csv").option("header","true").schema(descriptionSchema).load("/Users/yiwang/Documents/YiWang/Ebiz/Task 15/product_descriptions.csv")
title= title.drop(title.id)
title.show(10)
description.show(10)


attr.createOrReplaceTempView("attr")
#get brand, color and material
brand = sql_sc.sql("SELECT pid,value as brand from attr where name = 'MFG Brand Name'")
material = sql_sc.sql("SELECT pid,value as material from attr where name = 'Material'")
color = sql_sc.sql("SELECT pid,value as color from attr where name = 'Color Family'")

#result=train.union(test)
title=title.join(description, title.pid == description.pid, "left").drop(description.pid)
title=title.join(brand, title.pid == brand.pid, "left").drop(brand.pid)
title=title.join(material, title.pid== material.pid,"left").drop(material.pid)
title=title.join(color, title.pid == color.pid,"left").drop(color.pid)

title.show(10)


# In[84]:

from pyspark.sql.functions import col, when
title=title.withColumn(
    "color", when(col("color").isNull(), "empty").otherwise(col("color")))
title=title.withColumn(
    "brand", when(col("brand").isNull(), "empty").otherwise(col("brand")))
title=title.withColumn(
    "material", when(col("material").isNull(), "empty").otherwise(col("material")))
title=title.withColumn(
    "description", when(col("description").isNull(), "empty").otherwise(col("description")))


title.show(10)


# In[85]:

from pyspark.sql import functions as sf
title = title.withColumn('joined_column', 
                    sf.concat( sf.col('description'),sf.lit('_'), sf.col('title'),sf.lit('_'), sf.col('brand'), sf.lit('_'), sf.col('material'),sf.lit('_'), sf.col('color')))
title = title.drop(title.title).drop(title.description).drop(title.brand).drop(title.material).drop(title.color)
title.show(10)


# In[93]:

from pyspark.ml.feature import HashingTF, IDF, Tokenizer
#tokenize terms
tokenizer = Tokenizer(inputCol="term", outputCol="term_words")
temp = tokenizer.transform(title)

hashingTF = HashingTF(inputCol="term_words", outputCol="rawFeatures", numFeatures=10)
temp = hashingTF.transform(temp)

idf = IDF(inputCol = "rawFeatures", outputCol="term_idf")
idfModel = idf.fit(temp)
temp = idfModel.transform(temp)
temp=temp.drop("rawFeatures")

#tokenize joined_column
tokenizer = Tokenizer(inputCol="joined_column", outputCol="joined_words")
temp = tokenizer.transform(temp)

hashingTF = HashingTF(inputCol="joined_words", outputCol="rawFeatures", numFeatures=10)
temp = hashingTF.transform(temp)

idf = IDF(inputCol = "rawFeatures", outputCol="joined_idf")
idfModel = idf.fit(temp)
temp = idfModel.transform(temp)
temp=temp.drop("rawFeatures")

temp.show(5)


# In[94]:

result = temp
result.show(5)


# In[96]:

from pyspark.sql.types import *
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf

#match words and then proceed to jaccard_similarity_score
def countMatchedWords(joined, term):
    l1=len(joined)
    l2=len(term)
    match = 0
    for i in range(l1):
        for j in range(l2):
            if joined[i] == term[j]:
                match+=2
            elif joined[i] in term[j]:
                match+=1
            elif term[j] in joined[i]:
                match+=1
        return match
matchUDF=udf(countMatchedWords, IntegerType())

result=result.withColumn("match", matchUDF("joined_words", "term_words"))
result.show(5)


# In[97]:

from sklearn.metrics import jaccard_similarity_score
def jaccardSimilarity(term, joined):
    result=float(jaccard_similarity_score(term, joined))
    return result
jaccardUDF=udf(jaccardSimilarity, DoubleType())

result = result.withColumn("term_joined_jaccard", jaccardUDF("term", "joined_column"))
result.show(5)


# In[ ]:



