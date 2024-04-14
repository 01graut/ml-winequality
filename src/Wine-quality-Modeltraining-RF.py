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
    parser.add_argument('--save_model_path', required=True, help='S3/Local path where model will be saved')
    parser.add_argument('--enable_grid_search',required=False,help='Run Grid Search')
    args = parser.parse_args()

    # Initialize Spark Session
    spark = SparkSession.builder.appName("TrainTestWorkflow").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # Load the datasets
    # train_data_path = "/workspaces/ml-winequality/dataset/TrainingDataset.csv"
    # test_data_path = "/workspaces/ml-winequality/dataset/ValidationDataset.csv"

    train_data_path = args.train_dataset
    # test_data_path = args.test_dataset

    train_df = spark.read.csv(train_data_path,header=True, 
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
    # Based on the Correlation table dropping few columns
    train_df=train_df.drop("total sulfur dioxide","residual sugar")
    print(train_df.columns)
    print(train_df.head(10))

    # Used copilot how to get rid of quotes from colum header
    # new_column_names = [col_name.strip('"') for col_name in test_df.columns]
    # test_df = test_df.toDF(*new_column_names)

   # Split the dataframe into 70% training and 30% testing
    train_df, test_df = train_df.randomSplit([0.7, 0.3], seed=42)


    # Assemble features
    featureCols = train_df.columns[:-1]  # Assuming the last column is the label

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

    # Assume 'model' is your trained PipelineModel from the training script
    model_path = args.save_model_path
    model.write().overwrite().save(model_path + "/wine-prediction-RF" )
    scaler.write().overwrite().save(model_path + "/scalar-RF")
    print(f"Model saved to path : {model_path}")
    fitted_pipeline2.write().overwrite().save(model_path + "/wine-prediction-pipeline-RF")
    print("\nCompleted!!!")

    # #################### GRID SEARCH ####################
    if args.enable_grid_search:
        print("\nStarting Grid Search.....")
        # Define a parameter grid
        paramGrid = ParamGridBuilder() \
            .addGrid(model.numTrees, [10, 15, 20, 30, 40]) \
            .addGrid(model.maxDepth, [ 5, 10, 15, 20, 25]) \
            .build()

        # Configure CrossValidator
        crossval = CrossValidator(estimator=pipeline2,
                                estimatorParamMaps=paramGrid,
                                evaluator=MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1"),
                                    # evaluator=RegressionEvaluator(labelCol="quality"),
                                numFolds=10)

        # Run cross-validation, and choose the best set of parameters.
        cvModel = crossval.fit(train_df)

        # Make predictions on test data
        predictions = cvModel.transform(test_df)
        predictions = predictions.withColumn("prediction", round(col("prediction"),0).cast("double"))
        # Evaluate the best model
        evaluator = MulticlassClassificationEvaluator(labelCol="quality", predictionCol="prediction", metricName="f1")
        f1_score = evaluator.evaluate(predictions)

        print(f"Best Model F1 Score on Test Data using GridSearch: {f1_score}")
