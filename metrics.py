from sklearn.metrics import precision_score, recall_score, f1_score


def compute_metrics(preds, labels):
    preds = preds.numpy()
    labels = labels.numpy()

    results = {}

    for i, name in enumerate(["blur", "low_light"]):
        results[name] = {
            "precision": precision_score(labels[:, i], preds[:, i], zero_division=0),
            "recall": recall_score(labels[:, i], preds[:, i], zero_division=0),
            "f1": f1_score(labels[:, i], preds[:, i], zero_division=0)
        }

    return results