from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
from pyspark.ml.feature import VectorAssembler, MinMaxScaler
from pyspark.sql.functions import *
from pyspark.ml.classification import RandomForestClassificationModel

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Wine Quality Inference with PySpark')
    parser.add_argument('--model_path', required=True, help='Path to the trained model')
    parser.add_argument('--new_data', required=True, help='Path to the new data for inference')
    # parser.add_argument('--scaler_model_path', required=True, help='Path where the MinMaxScaler model was saved')
    args = parser.parse_args()

    # Initialize Spark Session
    spark = SparkSession.builder \
    .appName("WineQualityInference") \
    .config("spark.jars.packages", "org.apache.spark:spark-hadoop-cloud_2.12:3.3.0") \
    .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
    .getOrCreate()

    # Load the trained model
    # rfModel = RandomForestRegressor.load(args.model_path)
    # rfModel = RandomForestRegressor.load(args.model_path)
    pipelineModel = PipelineModel.load(args.model_path)

    # print(rfModel.featuresCol)

    # Load new data for inference
    new_data = spark.read.csv(args.new_data, header=True, inferSchema=True, sep=';', quote='"')

    # Correct for quotes in column names if necessary
    new_column_names = [col_name.strip('"') for col_name in new_data.columns]
    new_data = new_data.toDF(*new_column_names)
    print("Dropping feature columsn : \"total sulfur dioxide","residual sugar\"")
    new_data=new_data.drop("total sulfur dioxide","residual sugar")
    # print(new_data.columns)
    new_data.head(10)
    # # Assuming you know the feature columns used during training
    # feature_columns = new_data.columns[:-1]  # Excluding the target label if present

    # # Apply VectorAssembler to assemble features into a vector (same as during training)
    # assembler = VectorAssembler(inputCols=feature_columns, outputCol="assembledFeatures")
    # assembled_data = assembler.transform(new_data)

    # # Load the saved MinMaxScaler model and apply it to the assembled features
    # scalerModel = MinMaxScaler.load(args.scaler_model_path)
    # print(scalerModel.getInputCol)
    # print(scalerModel.getOutputCol)

    # scaled_data_fit = scalerModel.fit(assembled_data)
    # # scaled_data = scaled_data_fit.transform(assembled_data)

    # Make predictions with the RandomForest model on the scaled data
    # predictions = rfModel.transform(scaled_data_fit)

    predictions = pipelineModel.transform(new_data)
    
    predictions.withColumn("prediction", round(col("prediction"),0).cast("double")).show(10)
    predictions = predictions.withColumn("prediction", round(col("prediction"),0).cast("double"))

    # Evaluation on F1 score 
    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
    f1_score = evaluator.evaluate(predictions)

    # print("Using f1 score for evaluation ")
    # print(f"Best Model F1 Score on Test Data: {f1_score*100}%")
    # print("Best Model F1 Score on Test Data: {:.1f}%".format(f1_score*100) )

    # Evaluation on Accuracy score 
    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="accuracy")
    acc_score = evaluator.evaluate(predictions)

    # print("Using Accuracy score for evaluation ")
    # print(f"Model Accuracy Score on Test Data: {acc_score*100}%")
    # print("Model Accuracy Score on Test Data: {:.1f}%".format(acc_score*100))

    print("Model Score on Test Data: \nF1 Score \t=> {:.1f}% \nAccuracy Score  => {:.1f}%".format(f1_score*100,acc_score*100))

    # Stop the Spark session
    spark.stop()
