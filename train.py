from src.data.data_loader import DatasetLoader
from src.preprocessing.preprocess import Preprocessor, DatasetProcessor
from src.training.model import MLPModel
from src.training.trainer import Trainer

# Dataset Configuration
DATASET_URL = "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
LOCAL_PATH = "dataset/names.txt"
BLOCK_SIZE = 3
TRAIN_RATIO = 0.8
TEST_RATIO = 0.1

# Load dataset
dataset_loader = DatasetLoader(DATASET_URL, LOCAL_PATH)
dataset_loader.download_dataset()
words = dataset_loader.read_dataset()

if not words:
    raise RuntimeError("Failed to load dataset. Please check the file path.")

# Preprocess dataset
preprocessor = Preprocessor(words)
dataset_processor = DatasetProcessor(words, preprocessor, BLOCK_SIZE)

# Split dataset
X_train, Y_train, X_dev, Y_dev, X_test, Y_test = dataset_processor.split_dataset(
    TRAIN_RATIO, TEST_RATIO
)

print("Dataset processing complete!")

# Initialize Model
vocab_size = len(preprocessor.stoi)
mlp_model = MLPModel(vocab_size, BLOCK_SIZE)

# Train Model
trainer = Trainer(mlp_model, X_train, Y_train)
trained_params = trainer.train()

print("Model training complete!")
