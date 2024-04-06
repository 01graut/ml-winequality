from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import MinMaxScaler, StandardScaler
from pyspark.sql.functions import *
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Wine Quality Classification with PySpark')
    parser.add_argument('--dataset', required=True, help='Path to the dataset')
    args = parser.parse_args()

    # Initialize Spark Session
    spark = SparkSession.builder.appName("WineQualityClassification").getOrCreate()

    # Load dataset
    train_df = spark.read.csv(args.dataset,header=True, 
                        inferSchema=True,
                        sep=';'
                        ,quote='"')
    
    # test_df = spark.read.csv(test_data_path,header=True, 
    #                     inferSchema=True,
    #                     sep=';'
    #                     ,quote='"')

    # Used copilot how to get rid of quotes from colum header
    new_column_names = [col_name.strip('"') for col_name in train_df.columns]
    train_df = train_df.toDF(*new_column_names)
    df=train_df
    # Assuming the last column is the label for classification and it's numerical
    # Convert it to categorical if necessary
    indexer = StringIndexer(inputCol=df.columns[-1], outputCol="label").fit(df)
    feature_columns = df.columns[:-1]  # Exclude the last column which is the label


    assembler = VectorAssembler(inputCols=feature_columns, outputCol="assembledFeatures")
    # Normalize features using Min-Max Scaling
    scaler = MinMaxScaler(inputCol="assembledFeatures", outputCol="features")
    # Split the data into training and test sets
    (trainingData, testData) = df.randomSplit([0.7, 0.3])


    
    # Define the model
    lr = LogisticRegression(featuresCol="features", labelCol="label"
                            ,elasticNetParam=1  # L2 regularization
                            ,regParam=0.01)  # Regularization strength)

    # Chain indexer and model in a Pipeline
    pipeline = Pipeline(stages=[indexer,assembler,scaler, lr])

    # Train model
    model = pipeline.fit(trainingData)

    # Make predictions
    predictions = model.transform(testData)
    predictions = predictions.withColumn("prediction", round(col("prediction"),0).cast("double"))
    # Select example rows to display
    predictions.select("prediction", "label", "features").show(5)

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    f1_score = evaluator.evaluate(predictions)
    print(f"F1 Score = {f1_score}")

    # Stop the Spark session
    spark.stop()
