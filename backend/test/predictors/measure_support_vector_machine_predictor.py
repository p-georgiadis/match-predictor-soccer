from unittest import TestCase

from matchpredictor.evaluation.evaluator import Evaluator
from matchpredictor.matchresults.results_provider import training_results, validation_results
from matchpredictor.predictors.support_vector_machine_predictor import train_svm_predictor
from test.predictors import csv_location


class TestSupportVectorMachinePredictor(TestCase):
    def test_accuracy_for_svm_predictor(self) -> None:
        # Load training and validation data
        training_data = training_results(csv_location, 2019, result_filter=lambda result: result.season >= 2015)
        validation_data = validation_results(csv_location, 2019)

        # Train the SVM predictor
        predictor = train_svm_predictor(training_data)

        # Measure accuracy using the Evaluator
        accuracy, _ = Evaluator(predictor).measure_accuracy(validation_data)

        # Assert that the accuracy is greater than or equal to a meaningful threshold
        threshold = 0.4  # Adjust this threshold based on your use case
        self.assertGreaterEqual(accuracy, threshold, f"Accuracy should be greater than or equal to {threshold}")
