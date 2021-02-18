# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
def predictionexample():
    # Use a breakpoint in the code line below to debug your script.
    spark = SparkSession.builder.appName('Customers').getOrCreate()
    dataset = spark.read.csv("Ecommerce_Customers.csv", inferSchema=True, header=True)
    #build feature using Vectorassembler
    featureassembler = VectorAssembler(inputCols=["Avg Session Length", "Time on App", "Time on Website", "Length of Membership"],outputCol="Independent Features")
    output = featureassembler.transform(dataset)
    output.show()

    finalized_data = output.select("Independent Features", "Yearly Amount Spent")
    finalized_data.show()
    #split the data 80%, 20%
    train_data, test_data = finalized_data.randomSplit([0.80, 0.20])
    regressor = LinearRegression(featuresCol='Independent Features', labelCol='Yearly Amount Spent')
    regressor = regressor.fit(train_data)
    pred_results = regressor.evaluate(test_data)
    pred_results.predictions.show(40)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    predictionexample()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
