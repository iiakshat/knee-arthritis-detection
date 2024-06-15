from cnnClassifier import logger
from cnnClassifier.pipeline.S1_data_ingestion import DataIngestionTrainingPipeline, STAGE1
from cnnClassifier.pipeline.S2_prepare_base_model import PrepareBaseModelTrainingPipeline, STAGE2


# Data Ingestion Stage
try:
    logger.info(f">>>>>> stage {STAGE1} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE1} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e

# Prepare Base Model Stage
try:
    logger.info(f">>>>>> stage {STAGE2} started <<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE2} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e
