from pathlib import Path
import os

if os.path.exists('/content/drive'):
    # Running in Colab - dataset on Drive
    BASE_PATH = Path('/content/drive/MyDrive/amazon_ml')
    DATA_MODE = 'colab'
    print(f"Colab mode - using Drive dataset")
else:
    BASE_PATH = Path.cwd() / 'amazon-ml-challenge-2026'
    DATA_MODE = 'local'
    print(f"Local mode - using validation artifacts only")

# Validation artifacts (same for both Colab and local)
ARTIFACT_DIR = BASE_PATH / 'code' /'business_entity_resolution' /'src' / 'validation_artifacts'

print(f"✅ BASE_PATH: {BASE_PATH}")
print(f"✅ ARTIFACT_DIR: {ARTIFACT_DIR}")