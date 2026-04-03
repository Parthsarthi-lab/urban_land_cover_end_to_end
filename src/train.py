import logging

from src.pipeline import TrainingPipeline


logging.basicConfig(
    filename="training_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
    results = TrainingPipeline().run()

    print("Selected feature count:", len(results["selected_features"]))
    print("Best Params:", results["model_selection_info"]["best_params"])
    print("Best Score:", results["model_selection_info"]["best_score"])
    print("Train F1:", results["evaluation_info"]["train_f1_weighted"])
    print("Test F1:", results["evaluation_info"]["test_f1_weighted"])
    print("Saved Model:", results["model_path"])


if __name__ == "__main__":
    main()
