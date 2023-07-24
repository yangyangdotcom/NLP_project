from pathlib import Path
import pytest
from config import config
from bnai import main, predict

@pytest.fixture(scope="module")
def artifacts():
    run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = main.load_artifacts(run_id=run_id)
    return artifacts

@pytest.mark.parametrize(
    "text, tag",
    [
        (
            "Transformers applied to NLP have revolutionized machine learning.",
            "natural-language-processing",
        ),
        (
            "Transformers applied to NLP have disrupted machine learning.",
            "natural-language-processing",
        ),
    ],
)
def test_inv(text_a, text_b, tag, artifacts):
    """INVariance via verb injection (changes should not affect outputs)"""
    predicted_tag = predict.predict(texts=[text], artifacts=artifacts)[0]["predicted_tag"]
    assert tag == predicted_tag

@pytest.mark.parametrize(
    "text, tag",
    [
        (
            "ML applied to text classification.",
            "natural-language-processing",
        ),
        (
            "ML applied to image classification.",
            "computer-vision",
        ),
        (
            "CNNs for text classification.",
            "natural-language-processing",
        )
    ],
)
def test_dir(text, tag, artifacts):
    """DIRectional expectations (changes with known outputs)."""
    predicted_tag = predict.predict(texts=[text], artifacts=artifacts)[0]["predicted_tag"]
    assert tag == predicted_tag

@pytest.mark.parametrize(
    "text, tag",
    [
        (
            "Natural language processing is the next big wave in machine learning.",
            "natural-language-processing",
        ),
        (
            "MLOps is the next big wave in machine learning.",
            "mlops",
        ),
        (
            "This should not produce any relevant topics.",
            "other",
        ),
    ],
)
def test_mft(text, tag, artifacts):
    """Minimum Functionality Tests (simple input/output pairs)."""
    predicted_tag = predict.predict(texts=[text], artifacts=artifacts)[0]["predicted_tag"]
    assert tag == predicted_tag