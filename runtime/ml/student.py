"""Target-conditioned six-class student and its evaluation metrics."""

import time

import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score,
)

from lib.sentiment import CHOICES, OFF_TOPIC, SCORE_LEVELS


SCORES = np.asarray(SCORE_LEVELS)


def normalized_distribution(values):
    distribution = np.asarray(values, dtype=np.float64)
    if distribution.shape != (len(CHOICES),) or not np.isfinite(distribution).all():
        raise ValueError("Expected six finite class probabilities")
    if (distribution < 0).any() or distribution.sum() <= 0:
        raise ValueError("Class probabilities must be nonnegative and have positive mass")
    return distribution / distribution.sum()


def split_threads(items, validation_fraction):
    latest = {}
    for item in items:
        story_id = item["story_id"]
        latest[story_id] = max(latest.get(story_id, ""), item["published_at"])
    if len(latest) < 2:
        raise ValueError("Training needs comments from at least two distinct threads")
    groups = sorted(latest, key=lambda story: (latest[story], story))
    holdout_size = min(len(groups) - 1, max(1, round(len(groups) * validation_fraction)))
    held_out = set(groups[-holdout_size:])
    train = [i for i, item in enumerate(items) if item["story_id"] not in held_out]
    test = [i for i, item in enumerate(items) if item["story_id"] in held_out]
    if len(train) < 2:
        raise ValueError("The training split needs at least two comments")
    return train, test


def fit(items, targets, kind, config, prior=None, on_epoch=None):
    if kind != "encoder":
        raise ValueError("The supported student kind is encoder")
    if not items or len(items) != len(targets):
        raise ValueError("Training needs one teacher distribution per comment")
    statistic = config.get("score_statistic", "mean")
    if statistic not in {"mean", "median"}:
        raise ValueError("score_statistic must be mean or median")
    import neural

    labels = np.stack([normalized_distribution(target) for target in targets]).astype(np.float32)
    artifact, optimization = neural.fit(items, labels, config, prior, on_epoch=on_epoch)
    artifact["score_statistic"] = statistic
    optimization.update(training_examples=len(items), training_items=len({str(item["id"]) for item in items}))
    return artifact, optimization


def predict(artifact, items):
    if artifact.get("kind") != "encoder":
        raise ValueError("This artifact is not a supported encoder student")
    if not items:
        return np.empty((0, len(CHOICES)), dtype=np.float64)
    import neural

    return neural.predict(artifact, items)


def relevant_score(distribution, statistic="mean"):
    sentiment = np.asarray(distribution[:OFF_TOPIC], dtype=np.float64)
    mass = sentiment.sum()
    if mass <= 0:
        return None
    if statistic == "median":
        index = min(int(np.searchsorted(np.cumsum(sentiment), mass / 2)), len(SCORES) - 1)
        return float(SCORES[index])
    return float(np.dot(SCORES, sentiment) / mass)


def score_distribution(distribution, statistic="mean"):
    distribution = normalized_distribution(distribution)
    relevant = bool(distribution[OFF_TOPIC] < 0.5)
    return (relevant_score(distribution, statistic) if relevant else None), relevant


def predictions(artifact, items):
    result = []
    for item, distribution in zip(items, predict(artifact, items)):
        score, relevant = score_distribution(distribution, artifact.get("score_statistic", "mean"))
        prediction = {
            "comment_id": item["id"],
            "probabilities": [float(value) for value in distribution],
            "score": score,
            "relevant": relevant,
        }
        if item.get("content_hash"):
            prediction["content_hash"] = item["content_hash"]
        result.append(prediction)
    return result


def fidelity(expected, actual):
    target = np.asarray(expected, dtype=np.float64)
    observed = np.asarray(actual, dtype=np.float64)
    if target.shape != observed.shape:
        raise ValueError("Reference and prediction counts must match")
    if len(target) == 0:
        return {"kl_divergence": None, "class_agreement": None}
    target = np.stack([normalized_distribution(row) for row in target])
    target_log = np.log(np.clip(target, 1e-12, 1.0))
    observed_log = np.log(np.clip(observed, 1e-12, 1.0))
    return {
        "kl_divergence": float(np.sum(target * (target_log - observed_log), axis=1).mean()),
        "class_agreement": float(np.mean(target.argmax(axis=1) == observed.argmax(axis=1))),
    }


def label_metrics(labels, probabilities, statistic="mean"):
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) == 0 or len(labels) != len(probabilities) or ((labels < 0) | (labels >= len(CHOICES))).any():
        raise ValueError("Quality evaluation needs matching reference labels in the six-class range")
    probabilities = np.stack([normalized_distribution(row) for row in probabilities])
    actual = probabilities.argmax(axis=1)
    expected_relevance = labels != OFF_TOPIC
    actual_relevance = probabilities[:, OFF_TOPIC] < 0.5
    relevant_indices = np.flatnonzero(expected_relevance)
    score_errors = []
    for index in relevant_indices:
        score = relevant_score(probabilities[index], statistic)
        if actual_relevance[index] and score is not None:
            score_errors.append(abs(float(SCORES[labels[index]]) - score))
    sentiment = np.column_stack([
        probabilities[:, :2].sum(axis=1), probabilities[:, 2], probabilities[:, 3:OFF_TOPIC].sum(axis=1),
    ]).argmax(axis=1)
    polarity = np.asarray([0, 0, 1, 2, 2, 3])
    entity_expected = polarity[labels]
    entity_actual = np.where(actual_relevance, sentiment, 3)
    expected_polarity = entity_expected[relevant_indices]
    actual_polarity = entity_actual[relevant_indices]
    return {
        "items": len(labels),
        "score_statistic": statistic,
        "relevance_definition": "Relevant if P(F) < 0.5; class accuracy independently uses six-class argmax",
        "relevance_accuracy": float(accuracy_score(expected_relevance, actual_relevance)),
        "relevance_macro_f1": float(f1_score(expected_relevance, actual_relevance, labels=[False, True], average="macro", zero_division=0)),
        "relevance_precision": float(precision_score(expected_relevance, actual_relevance, zero_division=0)),
        "relevance_recall": float(recall_score(expected_relevance, actual_relevance, zero_division=0)),
        "score_mae": float(np.mean(score_errors)) if score_errors else None,
        "score_mae_scope": "True relevant comments accepted by the classifier; rejected comments count as relevance and polarity errors",
        "scored_items": len(score_errors),
        "unscored_relevant_items": len(relevant_indices) - len(score_errors),
        "class_accuracy": float(accuracy_score(labels, actual)),
        "class_macro_f1": float(f1_score(labels, actual, labels=list(range(len(CHOICES))), average="macro", zero_division=0)),
        "present_class_macro_f1": float(f1_score(labels, actual, labels=sorted(set(labels.tolist())), average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, actual, labels=list(range(len(CHOICES)))).tolist(),
        "per_class": classification_report(labels, actual, labels=list(range(len(CHOICES))), target_names=list(CHOICES), zero_division=0, output_dict=True),
        "sentiment_polarity_items": len(relevant_indices),
        "sentiment_polarity_accuracy": float(accuracy_score(expected_polarity, actual_polarity)) if len(relevant_indices) else None,
        "sentiment_polarity_macro_f1": float(f1_score(expected_polarity, actual_polarity, labels=[0, 1, 2], average="macro", zero_division=0)) if len(relevant_indices) else None,
        "polarity_definition": "Summed negative/neutral/positive probabilities; a predicted F counts as an error on relevant comments",
        "entity_polarity_accuracy": float(accuracy_score(entity_expected, entity_actual)),
        "entity_polarity_macro_f1": float(f1_score(entity_expected, entity_actual, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
    }


def evaluate(artifact, items, teacher_probabilities=None):
    started = time.perf_counter()
    probabilities = predict(artifact, items)
    elapsed = time.perf_counter() - started
    report = {
        "items": len(items),
        "threads": len({item["story_id"] for item in items}),
        "decision_rule": artifact["decision_rule"],
        "student_ms_per_item": elapsed * 1000 / len(items) if items else None,
        "timing_note": "All comments run through the neural student; includes model loading on a cold process",
    }
    if teacher_probabilities is not None:
        report["teacher_fidelity"] = fidelity(teacher_probabilities, probabilities)
    indices = [index for index, item in enumerate(items) if item.get("human_label") is not None]
    if indices:
        labels = [items[index]["human_label"] for index in indices]
        report["human"] = label_metrics(labels, probabilities[indices], artifact.get("score_statistic", "mean"))
        if teacher_probabilities is not None:
            report["teacher_human"] = label_metrics(labels, [teacher_probabilities[index] for index in indices], artifact.get("score_statistic", "mean"))
    return report
