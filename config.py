import os

class Config:
    # Paths
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
    RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Data
    IMG_SIZE = 128
    BATCH_SIZE = 8
    NUM_WORKERS = 4

    # Train/val/test ratios
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    SEED = 42

    # Model
    SPATIAL_BACKBONE = "resnet50"
    FREQ_BACKBONE = "resnet18"
    FUSION_HIDDEN_DIM = 512
    DROPOUT = 0.5

    # Training
    NUM_EPOCHS = 10
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    EARLY_STOPPING_PATIENCE = 7

    # Logging / checkpointing
    BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_fusion_model.pth")


cfg = Config()
