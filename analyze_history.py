"""
analyze_history.py
------------------
Reads a Keras training_history.json and tells you whether the run is
UNDERFIT, GOOD FIT, OVERFIT, or "TOO GOOD (leakage-suspect)".

Usage:
    python analyze_history.py                       # uses model/training_history.json
    python analyze_history.py path/to/history.json
"""

import sys
import os
import json

HISTORY_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join("model", "training_history.json")


def load(path):
    with open(path) as f:
        return json.load(f)


def smooth_last(values, k=3):
    """Average of the last k values (robust to single-epoch noise)."""
    vals = values[-k:] if len(values) >= k else values
    return sum(vals) / len(vals)


def argmin(values):
    return min(range(len(values)), key=lambda i: values[i])


def diagnose(h):
    acc = h.get('accuracy', [])
    val_acc = h.get('val_accuracy', [])
    loss = h.get('loss', [])
    val_loss = h.get('val_loss', [])

    if not acc or not val_acc:
        return "No accuracy data found in history.", {}

    final_train = smooth_last(acc)
    final_val = smooth_last(val_acc)
    best_val = max(val_acc)
    best_val_epoch = val_acc.index(best_val) + 1
    gap = final_train - final_val

    # Did val_loss bottom out and then rise again? (classic overfitting onset)
    val_loss_min_epoch = argmin(val_loss) + 1 if val_loss else None
    val_loss_rose = False
    if val_loss:
        rose_amount = smooth_last(val_loss) - min(val_loss)
        val_loss_rose = rose_amount > 0.03  # meaningful rebound

    metrics = {
        'epochs': len(acc),
        'final_train_acc': round(final_train, 4),
        'final_val_acc': round(final_val, 4),
        'best_val_acc': round(best_val, 4),
        'best_val_epoch': best_val_epoch,
        'train_val_gap': round(gap, 4),
        'val_loss_min_epoch': val_loss_min_epoch,
        'val_loss_rebounded': val_loss_rose,
    }

    # --- classification rules ---
    if final_train < 0.75 and final_val < 0.75:
        verdict = "UNDERFIT"
        why = ("Both training and validation accuracy are low. The model has not "
               "learned the task yet. Train longer, increase model capacity / unfreeze "
               "more layers, raise the learning rate, or check that labels are correct.")
    elif gap > 0.15 and val_loss_rose:
        verdict = "OVERFIT"
        why = (f"Training accuracy ({final_train:.1%}) is much higher than validation "
               f"({final_val:.1%}) and val_loss bottomed at epoch {val_loss_min_epoch} "
               "then climbed. The model is memorizing the training set. Add augmentation, "
               "more dropout/regularization, more data, or stop earlier "
               f"(best val was epoch {best_val_epoch}).")
    elif gap > 0.15:
        verdict = "MILD OVERFIT"
        why = (f"Train-val gap is {gap:.1%}. Some memorization, but val_loss hasn't "
               "clearly rebounded. Consider a bit more augmentation/regularization.")
    elif final_val >= 0.85 and gap <= 0.10:
        verdict = "GOOD FIT (on paper)  --  CHECK FOR LEAKAGE"
        why = (f"Validation accuracy is high ({final_val:.1%}) with a small gap ({gap:.1%}). "
               "That looks healthy ON THIS SPLIT. But if real-world performance is much "
               "worse, the validation set is not representative -- usually background "
               "shortcut learning or near-duplicate images shared between train/val. "
               "Re-check with a leakage-free split and a cropped pipeline.")
    else:
        verdict = "REASONABLE FIT"
        why = (f"Validation accuracy {final_val:.1%} with gap {gap:.1%}. Room to improve "
               "but no severe over/underfitting signature.")

    return verdict, metrics, why


def maybe_plot(h, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib not available -- skipping plot)")
        return

    epochs = range(1, len(h['accuracy']) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, h['accuracy'], label='train')
    ax[0].plot(epochs, h['val_accuracy'], label='val')
    ax[0].set_title('Accuracy'); ax[0].set_xlabel('epoch'); ax[0].legend(); ax[0].grid(alpha=.3)
    if 'loss' in h:
        ax[1].plot(epochs, h['loss'], label='train')
        ax[1].plot(epochs, h['val_loss'], label='val')
        ax[1].set_title('Loss'); ax[1].set_xlabel('epoch'); ax[1].legend(); ax[1].grid(alpha=.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout(); fig.savefig(out_path, dpi=110)
    print(f"Saved plot -> {out_path}")


def main():
    if not os.path.exists(HISTORY_PATH):
        print(f"History file not found: {HISTORY_PATH}")
        raise SystemExit(1)

    h = load(HISTORY_PATH)
    verdict, metrics, why = diagnose(h)

    print("=" * 64)
    print(f"TRAINING HISTORY ANALYSIS  ({HISTORY_PATH})")
    print("=" * 64)
    for k, v in metrics.items():
        print(f"  {k:<22}: {v}")
    print("-" * 64)
    print(f"  VERDICT: {verdict}")
    print("-" * 64)
    print(why)
    print("=" * 64)

    maybe_plot(h, os.path.join("model", "plots", "history_analysis.png"))


if __name__ == "__main__":
    main()
