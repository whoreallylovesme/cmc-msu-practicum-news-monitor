"""A small, entity-specific classifier trained against teacher distributions."""

import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import FeatureUnion

from lib.sentiment import CHOICES, OFF_TOPIC, SCORE_LEVELS
from lib.mentions import MentionMatcher


SCORES = np.asarray(SCORE_LEVELS)


def normalized_distribution(values):
    distribution = np.asarray(values, dtype=np.float64)
    if distribution.shape != (len(CHOICES),) or not np.isfinite(distribution).all():
        raise ValueError("Expected six finite class probabilities")
    if (distribution < 0).any() or distribution.sum() <= 0:
        raise ValueError("Class probabilities must be nonnegative and have positive mass")
    return distribution / distribution.sum()


def context(item):
    return {
        "story_title": str(item.get("story_title") or "")[:320],
        "parent_text": str(item.get("parent_text") or "")[:480],
        "comment": str(item.get("text") or "")[:1200],
    }


def document(item):
    fields = context(item)
    return " ".join(f"{name} {value}" for name, value in fields.items())


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


def make_vectorizer(config):
    feature_budget = int(config["max_features"])
    word_features = feature_budget // 2
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(ngram_range=(1, 2), max_features=word_features, sublinear_tf=True),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=feature_budget - word_features,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def fit(items, targets, kind, config, prior=None):
    policy = {
        "entity": config.get("semantic_entity", ""),
        "aliases": config.get("semantic_aliases", []),
        "options": config.get("semantic_matching", {}),
    }
    if prior is not None and prior.get("mention_policy") not in (None, policy):
        raise ValueError("Mention matching policy changed; train from scratch to apply the new scope")
    indices = candidate_indices({"mention_policy": policy}, items) if policy["entity"] else list(range(len(items)))
    if not indices:
        raise ValueError("Training needs clear entity mentions; collect matching comments first")
    artifact, optimization = _fit([items[index] for index in indices], [targets[index] for index in indices], kind, config, prior)
    statistic = config.get("score_statistic", "mean")
    if statistic not in {"mean", "median"}:
        raise ValueError("score_statistic must be mean or median")
    artifact["score_statistic"] = statistic
    optimization.update(training_candidates=len({str(items[index]["id"]) for index in indices}),
                        training_examples=len(indices), scope_excluded=len(items) - len(indices))
    if policy["entity"]:
        artifact["mention_policy"] = policy
    return artifact, optimization


def _fit(items, targets, kind, config, prior=None):
    if kind == "semantic":
        return fit_semantic(items, targets, config, prior)
    documents = [document(item) for item in items]
    if prior is None:
        vectorizer = make_vectorizer(config)
        features = vectorizer.fit_transform(documents).tocsr()
    else:
        vectorizer = prior["vectorizer"]
        features = vectorizer.transform(documents).tocsr()
    rows, columns = features.shape
    classes = len(CHOICES)
    labels = np.stack([normalized_distribution(target) for target in targets])
    if kind == "hard_label":
        labels = np.eye(classes)[labels.argmax(axis=1)]

    regularization = float(config["regularization"])
    hidden = 0
    if kind == "tiny_mlp":
        hidden = prior["hidden_bias"].size if prior is not None else int(config["hidden_units"])
    hidden_size = (columns + 1) * hidden
    output_size = ((hidden or columns) + 1) * classes

    def layers(parameters):
        first = parameters[:hidden_size].reshape(columns + 1, hidden) if hidden else None
        output = parameters[hidden_size:].reshape((hidden or columns) + 1, classes)
        return first, output

    initial = np.zeros(hidden_size + output_size, dtype=np.float64)
    first, output = layers(initial)
    if prior is not None:
        output[:-1], output[-1] = prior["weights"], prior["bias"]
        if first is not None:
            first[:-1], first[-1] = prior["hidden_weights"], prior["hidden_bias"]
    elif first is not None:
        rng = np.random.default_rng(42)
        first[:-1] = rng.normal(0, 0.05, size=first[:-1].shape)
        output[:-1] = rng.normal(0, 0.12, size=output[:-1].shape)

    def loss_gradient(parameters):
        first, output = layers(parameters)
        hidden_values = np.tanh(features @ first[:-1] + first[-1]) if first is not None else features
        logits = hidden_values @ output[:-1] + output[-1]
        log_predictions = logits - logsumexp(logits, axis=1, keepdims=True)
        loss = -(labels * log_predictions).sum() / rows
        loss += regularization * np.square(output[:-1]).sum() / 2
        residual = (np.exp(log_predictions) - labels) / rows
        output_gradient = np.vstack(
            [hidden_values.T @ residual + regularization * output[:-1], residual.sum(axis=0)]
        )
        gradient = output_gradient.ravel()
        if first is not None:
            loss += regularization * np.square(first[:-1]).sum() / 2
            backpropagated = (residual @ output[:-1].T) * (1 - np.square(hidden_values))
            first_gradient = np.vstack(
                [features.T @ backpropagated + regularization * first[:-1], backpropagated.sum(axis=0)]
            )
            gradient = np.concatenate([first_gradient.ravel(), gradient])

        if prior is not None:
            anchor = float(config.get("incremental_anchor", 0.001))
            change = parameters - initial
            loss += anchor * np.dot(change, change) / 2
            gradient += anchor * change
        return float(loss), gradient

    iteration_limit = config["max_iterations"] if prior is None else config.get("incremental_iterations", 35)
    result = minimize(
        loss_gradient,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(iteration_limit)},
    )
    if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise ValueError("Student optimization produced non-finite values")
    first, output = layers(result.x.astype(np.float32))
    artifact = {
        "vectorizer": vectorizer,
        "weights": output[:-1],
        "bias": output[-1],
    }
    if first is not None:
        artifact.update(hidden_weights=first[:-1], hidden_bias=first[-1])
    optimization = {
        "features": columns,
        "hidden_units": hidden,
        "parameters": int(result.x.size),
        "iterations": int(result.nit),
        "converged": bool(result.success),
        "objective": float(result.fun),
        "warm_started": prior is not None,
        "vocabulary_frozen": prior is not None,
    }
    return artifact, optimization


def fit_semantic(items, targets, config, prior=None):
    """Fit joint heads, or calibrate a frozen encoder's fixed polarity logits."""
    head = config.get("semantic_head", "joint")
    if head not in {"joint", "temperature"}:
        raise ValueError("semantic_head must be joint or temperature")
    if prior is not None and prior.get("head", "joint") != head:
        raise ValueError("Semantic head changed; train from scratch before changing semantic_head")
    if head == "temperature":
        return fit_temperature(items, targets, config, prior)
    import encoder

    entity = prior["entity"] if prior is not None else str(config.get("semantic_entity") or "").strip()
    if not entity:
        raise ValueError("Semantic student training requires its monitored entity")
    bundle = prior["encoder"] if prior is not None else encoder.prepare(config)
    aliases = prior["aliases"] if prior is not None else list(config.get("semantic_aliases", []))
    relevance_features, sentiment_features = encoder.features(bundle, items, entity, aliases)
    rows, columns = relevance_features.shape
    labels = np.stack([normalized_distribution(target) for target in targets])
    relevant_mass = labels[:, :OFF_TOPIC].sum(axis=1)
    stride = columns + 1
    initial = np.zeros(stride * (OFF_TOPIC + 1), dtype=np.float64)

    def layers(parameters):
        relevance = parameters[:stride]
        sentiment = parameters[stride:].reshape(stride, OFF_TOPIC)
        return relevance, sentiment

    relevance, sentiment = layers(initial)
    if prior is not None:
        relevance[:-1], relevance[-1] = prior["relevance_weights"], prior["relevance_bias"]
        sentiment[:-1], sentiment[-1] = prior["weights"], prior["bias"]
    regularization = float(config.get("semantic_regularization", 0.001))

    def loss_gradient(parameters):
        relevance, sentiment = layers(parameters)
        logits = relevance_features @ relevance[:-1] + relevance[-1]
        sentiment_logits = sentiment_features @ sentiment[:-1] + sentiment[-1]
        log_sentiment = sentiment_logits - logsumexp(sentiment_logits, axis=1, keepdims=True)
        # This factorization is exactly the cross entropy of the six-class joint
        # distribution. Off-topic examples contribute no sentiment target loss.
        loss = (np.logaddexp(0, logits) - relevant_mass * logits).mean()
        loss -= (labels[:, :OFF_TOPIC] * log_sentiment).sum() / rows
        loss += regularization * (np.square(relevance[:-1]).sum() + np.square(sentiment[:-1]).sum()) / 2
        relevance_residual = (expit(logits) - relevant_mass) / rows
        relevance_gradient = np.r_[
            relevance_features.T @ relevance_residual + regularization * relevance[:-1],
            relevance_residual.sum(),
        ]
        sentiment_residual = (np.exp(log_sentiment) * relevant_mass[:, None] - labels[:, :OFF_TOPIC]) / rows
        sentiment_gradient = np.vstack([
            sentiment_features.T @ sentiment_residual + regularization * sentiment[:-1],
            sentiment_residual.sum(axis=0),
        ])
        gradient = np.r_[relevance_gradient, sentiment_gradient.ravel()]
        if prior is not None:
            anchor = float(config.get("incremental_anchor", 0.001))
            change = parameters - initial
            loss += anchor * np.dot(change, change) / 2
            gradient += anchor * change
        return float(loss), gradient

    result = minimize(
        loss_gradient, initial, method="L-BFGS-B", jac=True,
        options={"maxiter": int(config.get("max_iterations", 160) if prior is None else
                                config.get("incremental_iterations", 35))},
    )
    if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise ValueError("Semantic student optimization produced non-finite values")
    relevance, sentiment = layers(result.x.astype(np.float32))
    artifact = {
        "kind": "semantic", "entity": entity, "aliases": aliases, "encoder": bundle,
        "relevance_weights": relevance[:-1], "relevance_bias": relevance[-1],
        "weights": sentiment[:-1], "bias": sentiment[-1],
    }
    optimization = {
        "features": columns, "hidden_units": 0, "parameters": int(result.x.size),
        "iterations": int(result.nit), "converged": bool(result.success), "objective": float(result.fun),
        "warm_started": prior is not None, "encoder_frozen": True,
        "encoder_sha256": bundle["model_sha256"],
        "encoder_bytes": len(bundle["model_bytes"]) if "model_bytes" in bundle else bundle["model_size_bytes"],
        "factorization": "binary relevance and five-class conditional sentiment",
    }
    return artifact, optimization


def _polarity_logits(features):
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 3 or not np.isfinite(values[:, :3]).all():
        raise ValueError("Temperature calibration requires three finite raw polarity logits")
    return values[:, :3]


def fit_temperature(items, targets, config, prior=None):
    """Calibrate three fixed polarity logits; learn no class or relevance bias."""
    import encoder

    entity = prior["entity"] if prior is not None else str(config.get("semantic_entity") or "").strip()
    if not entity:
        raise ValueError("Semantic student training requires its monitored entity")
    bundle = prior["encoder"] if prior is not None else encoder.prepare(config)
    if bundle.get("output_mode") not in {"nli", "sentiment_logits"}:
        raise ValueError("Temperature calibration requires a raw-logit encoder")
    aliases = prior["aliases"] if prior is not None else list(config.get("semantic_aliases", []))
    _, sentiment_features = encoder.features(bundle, items, entity, aliases)
    logits = _polarity_logits(sentiment_features)
    labels = np.stack([normalized_distribution(target) for target in targets])
    relevant = labels[:, OFF_TOPIC] < 0.5
    if not relevant.any():
        raise ValueError("Temperature calibration needs at least one relevant training target")
    collapsed = np.column_stack([
        labels[:, :2].sum(axis=1), labels[:, 2], labels[:, 3:OFF_TOPIC].sum(axis=1),
    ])[relevant]
    # Preserve the supplied smoothing, conditional on true relevance. Off-topic
    # rows never adjust the temperature or create a learned relevance function.
    collapsed /= collapsed.sum(axis=1, keepdims=True)
    logits = logits[relevant]
    initial = float(prior["log_temperature"]) if prior is not None else 0.0
    if not np.isfinite(initial) or not -4 <= initial <= 4:
        raise ValueError("Stored log_temperature must be finite and between -4 and 4")
    anchor = float(config.get("incremental_anchor", 0.001)) if prior is not None else 0.0
    if not np.isfinite(anchor) or anchor < 0:
        raise ValueError("incremental_anchor must be finite and nonnegative")

    def cross_entropy(log_temperature):
        scaled = logits * np.exp(-log_temperature)
        log_probabilities = scaled - logsumexp(scaled, axis=1, keepdims=True)
        return float(-(collapsed * log_probabilities).sum(axis=1).mean())

    def loss_gradient(parameters):
        log_temperature = float(parameters[0])
        scaled = logits * np.exp(-log_temperature)
        log_probabilities = scaled - logsumexp(scaled, axis=1, keepdims=True)
        change = log_temperature - initial
        loss = -(collapsed * log_probabilities).sum(axis=1).mean() + anchor * change * change / 2
        gradient = ((collapsed - np.exp(log_probabilities)) * scaled).sum(axis=1).mean()
        return float(loss), np.asarray([gradient + anchor * change], dtype=np.float64)

    result = minimize(
        loss_gradient, np.asarray([initial]), method="L-BFGS-B", jac=True,
        bounds=[(-4.0, 4.0)],
        options={"maxiter": int(config.get("incremental_iterations", 35) if prior is not None else
                                config.get("max_iterations", 160)), "gtol": 1e-8, "ftol": 1e-12},
    )
    if not np.isfinite(result.x).all() or not np.isfinite(result.fun):
        raise ValueError("Temperature calibration produced non-finite values")
    log_temperature = float(result.x[0])
    artifact = {
        "kind": "semantic", "head": "temperature", "entity": entity, "aliases": aliases,
        "encoder": bundle, "log_temperature": log_temperature,
    }
    optimization = {
        "features": 3, "hidden_units": 0, "parameters": 1,
        "head": "temperature",
        "output_semantics": "three-polarity sentiment index; A/C/E are transport slots, not five intensity predictions",
        "iterations": int(result.nit), "converged": bool(result.success), "objective": float(result.fun),
        "warm_started": prior is not None, "encoder_frozen": True,
        "encoder_sha256": bundle["model_sha256"],
        "encoder_bytes": len(bundle["model_bytes"]) if "model_bytes" in bundle else bundle["model_size_bytes"],
        "factorization": "mention gate and temperature-calibrated three-class polarity",
        "temperature": float(np.exp(log_temperature)), "log_temperature": log_temperature,
        "temperature_bounds": [float(np.exp(-4)), float(np.exp(4))],
        "training_relevant_examples": int(relevant.sum()),
        "training_off_topic_examples": int((~relevant).sum()),
        "cross_entropy_before": cross_entropy(initial),
        "cross_entropy_after": cross_entropy(log_temperature),
        "incremental_anchor": anchor,
    }
    return artifact, optimization


def predict(artifact, items):
    indices = candidate_indices(artifact, items)
    probabilities = np.zeros((len(items), len(CHOICES)), dtype=np.float64)
    probabilities[:, OFF_TOPIC] = 1
    if indices:
        probabilities[indices] = _predict(artifact, [items[index] for index in indices])
    return probabilities


def candidate_indices(artifact, items):
    policy = artifact.get("mention_policy")
    if not policy:
        return list(range(len(items)))
    matcher = MentionMatcher(policy["entity"], policy["aliases"], policy["options"])
    return [index for index, item in enumerate(items) if matcher.match(item) is not None]


def _predict(artifact, items):
    if not items:
        return np.empty((0, len(CHOICES)))
    if artifact.get("kind") == "semantic":
        import encoder

        relevance_features, sentiment_features = encoder.features(
            artifact["encoder"], items, artifact["entity"], artifact["aliases"]
        )
        if artifact.get("head") == "temperature":
            log_temperature = float(artifact["log_temperature"])
            if not np.isfinite(log_temperature) or not -4 <= log_temperature <= 4:
                raise ValueError("Stored log_temperature must be finite and between -4 and 4")
            polarity = softmax(_polarity_logits(sentiment_features) * np.exp(-log_temperature), axis=1)
            probabilities = np.zeros((len(items), len(CHOICES)), dtype=np.float64)
            probabilities[:, [0, 2, 4]] = polarity
            return probabilities
        relevant_mass = expit(relevance_features @ artifact["relevance_weights"] + artifact["relevance_bias"])
        sentiment = softmax(sentiment_features @ artifact["weights"] + artifact["bias"], axis=1)
        return np.column_stack([sentiment * relevant_mass[:, None], 1 - relevant_mass])
    features = artifact["vectorizer"].transform([document(item) for item in items])
    if "hidden_weights" in artifact:
        features = np.tanh(features @ artifact["hidden_weights"] + artifact["hidden_bias"])
    return softmax(features @ artifact["weights"] + artifact["bias"], axis=1)


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
    target_log = np.log(np.clip(target, 1e-12, 1.0))
    observed_log = np.log(np.clip(observed, 1e-12, 1.0))
    return {
        "kl_divergence": float(np.sum(target * (target_log - observed_log), axis=1).mean()),
        "class_agreement": float(np.mean(target.argmax(axis=1) == observed.argmax(axis=1))),
    }


def label_metrics(labels, probabilities, statistic="mean"):
    if len(labels) == 0:
        raise ValueError("Quality evaluation needs at least one reference label")
    probabilities = np.stack([normalized_distribution(values) for values in probabilities])
    actual = probabilities.argmax(axis=1)
    errors = []
    expected_relevance = []
    actual_relevance = []
    for label, values in zip(labels, probabilities):
        distribution = normalized_distribution(values)
        expected_relevance.append(label != OFF_TOPIC)
        actual_relevance.append(score_distribution(distribution)[1])
        if label != OFF_TOPIC and actual_relevance[-1]:
            score = relevant_score(distribution, statistic)
            if score is not None:
                errors.append(abs(float(SCORES[label]) - score))
    relevant_indices = [index for index, label in enumerate(labels) if label != OFF_TOPIC]
    # Marginalize fine intensity classes before deciding the three polarities.
    # Picking a fine-class winner first incorrectly penalizes split negative mass.
    sentiment = np.column_stack([
        probabilities[:, :2].sum(axis=1), probabilities[:, 2], probabilities[:, 3:5].sum(axis=1),
    ]).argmax(axis=1)
    polarity = lambda label: 0 if label < 2 else 1 if label == 2 else 2
    expected_polarity = [polarity(labels[index]) for index in relevant_indices]
    actual_polarity = [int(sentiment[index]) if actual_relevance[index] else 3 for index in relevant_indices]
    entity_expected = [polarity(label) if label != OFF_TOPIC else 3 for label in labels]
    entity_actual = [int(sentiment[index]) if actual_relevance[index] else 3 for index in range(len(labels))]
    return {
        "items": len(labels),
        "score_statistic": statistic,
        "relevance_accuracy": float(np.mean(np.equal(expected_relevance, actual_relevance))),
        "relevance_macro_f1": float(
            f1_score(expected_relevance, actual_relevance, labels=[False, True], average="macro", zero_division=0)
        ),
        "score_mae": float(np.mean(errors)) if errors else None,
        "scored_items": len(errors),
        "unscored_relevant_items": len(relevant_indices) - len(errors),
        "polarity_definition": "argmax of summed negative, neutral and positive probabilities; rejection is an error",
        "relevance_precision": float(precision_score(expected_relevance, actual_relevance, zero_division=0)),
        "relevance_recall": float(recall_score(expected_relevance, actual_relevance, zero_division=0)),
        "class_accuracy": float(accuracy_score(labels, actual)),
        "class_macro_f1": float(f1_score(labels, actual, labels=list(range(len(CHOICES))), average="macro", zero_division=0)),
        "present_class_macro_f1": float(f1_score(labels, actual, labels=sorted(set(labels)), average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, actual, labels=list(range(len(CHOICES)))).tolist(),
        "per_class": classification_report(labels, actual, labels=list(range(len(CHOICES))), target_names=list(CHOICES), zero_division=0, output_dict=True),
        "sentiment_polarity_items": len(relevant_indices),
        "sentiment_polarity_accuracy": float(accuracy_score(expected_polarity, actual_polarity)) if relevant_indices else None,
        "sentiment_polarity_macro_f1": float(f1_score(expected_polarity, actual_polarity, labels=[0, 1, 2], average="macro", zero_division=0)) if relevant_indices else None,
        "entity_polarity_accuracy": float(accuracy_score(entity_expected, entity_actual)),
        "entity_polarity_macro_f1": float(f1_score(entity_expected, entity_actual, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
    }


def evaluate(artifact, items, teacher_probabilities=None):
    start = time.perf_counter()
    probabilities = predict(artifact, items)
    elapsed = time.perf_counter() - start
    candidates = candidate_indices(artifact, items)
    report = {
        "items": len(items),
        "fine_intensity_metrics_applicable": artifact.get("head") != "temperature",
        "threads": len({item["story_id"] for item in items}),
        "student_ms_per_item": elapsed * 1000 / len(items) if items else None,
        "candidate_items": len(candidates),
        "student_ms_per_candidate": elapsed * 1000 / len(candidates) if candidates else None,
        "timing_note": "Current prepared/in-process feature caches are included; this is not an uncached backbone benchmark.",
    }
    if teacher_probabilities is not None:
        report["teacher_fidelity"] = fidelity(teacher_probabilities, probabilities)
        if candidates:
            report["teacher_fidelity_candidates"] = fidelity(
                [teacher_probabilities[index] for index in candidates], probabilities[candidates],
            )

    selected = [
        (item["human_label"], probability)
        for item, probability in zip(items, probabilities)
        if item.get("human_label") is not None
    ]
    if selected:
        labels, values = zip(*selected)
        report["human"] = label_metrics(labels, values, artifact.get("score_statistic", "mean"))
        if teacher_probabilities is not None:
            reference = [
                probability
                for item, probability in zip(items, teacher_probabilities)
                if item.get("human_label") is not None
            ]
            report["teacher_human"] = label_metrics(labels, reference, artifact.get("score_statistic", "mean"))
    return report
