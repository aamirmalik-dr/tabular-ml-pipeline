"""Build and fit a pipeline from a typed config, then compare several models.

Run this from the repository root with no network access:

    python examples/build_pipeline.py

It loads the committed sample, assembles a preprocessing plus model pipeline from
a :class:`~tabml.config.PipelineConfig`, fits it, and prints the tuned multi-model
comparison table.
"""

from __future__ import annotations

from tabml import (
    PipelineConfig,
    build_pipeline,
    compare_models,
    load_sample,
    results_table,
    train_test_split_frame,
)


def main() -> None:
    data = load_sample()
    train, test = train_test_split_frame(data, seed=0)

    # A single pipeline from a typed config: median-impute and standardize numeric
    # columns, most-frequent-impute and one-hot encode categoricals, insert a LASSO
    # selection step, then a gradient-boosting classifier.
    config = PipelineConfig(model="gradient_boosting")
    pipe = build_pipeline(config)
    pipe.fit(train.X, train.y)
    print("Fitted pipeline steps:", [name for name, _ in pipe.steps])
    print("Held-out accuracy:", round(pipe.score(test.X, test.y), 4))

    # The tuned comparison grid-searches every model in the config and ranks them.
    results = compare_models(train, test, config=config)
    print("\n" + results_table(results))
    print("\nBest model:", results[0].name)


if __name__ == "__main__":
    main()
