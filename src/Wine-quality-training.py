from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import MinMaxScaler, StandardScaler
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import *
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Wine Quality Prediction with PySpark')
    parser.add_argument('--train_dataset', required=True, help='S3 path to the Training Dataset')
    parser.add_argument('--test_dataset', required=True, help='S3 path to the Validation Dataset')
    args = parser.parse_args()

    # Initialize Spark Session
    spark = SparkSession.builder.appName("TrainTestWorkflow").getOrCreate()

    # Load the datasets
    # train_data_path = "/workspaces/ml-winequality/dataset/TrainingDataset.csv"
    # test_data_path = "/workspaces/ml-winequality/dataset/ValidationDataset.csv"

    train_data_path = args.train_dataset
    test_data_path = args.test_dataset

    train_df = spark.read.csv(train_data_path,header=True, 
                        inferSchema=True,
                        sep=';'
                        ,quote='"')
    test_df = spark.read.csv(test_data_path,header=True, 
                        inferSchema=True,
                        sep=';'
                        ,quote='"')

    # Used copilot how to get rid of quotes from colum header
    new_column_names = [col_name.strip('"') for col_name in train_df.columns]
    train_df = train_df.toDF(*new_column_names)

    # Used copilot how to get rid of quotes from colum header
    new_column_names = [col_name.strip('"') for col_name in test_df.columns]
    test_df = test_df.toDF(*new_column_names)

    # Assemble features
    # Assemble features
    featureCols = train_df.columns[:-1]  # Assuming the last column is the label

    # MINMAXSCALAR

    # Assemble features
    assembler = VectorAssembler(inputCols=featureCols, outputCol="assembledFeatures")

    # Normalize features using Min-Max Scaling
    scaler = MinMaxScaler(inputCol="assembledFeatures", outputCol="normalizedFeatures")

    # Initialize the model
    model = RandomForestRegressor(featuresCol="normalizedFeatures", labelCol='quality',
                                maxDepth=20,
                                numTrees=25,
                                seed=42,
                                )


    # Create a Pipeline
    pipeline2 = Pipeline(stages=[assembler, scaler, model])

    # Train the model
    fitted_pipeline2 = pipeline2.fit(train_df)

    # Evaluate using MixMax Scalar


    # Make predictions on test data
    # Make predictions
    predictions = fitted_pipeline2.transform(test_df)

    predictions = predictions.withColumn("prediction", round(col("prediction"),0).cast("double"))
    # Evaluate the best model
    evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
    f1_score = evaluator.evaluate(predictions)

    print(f"Best Model F1 Score on Test Data: {f1_score}")
