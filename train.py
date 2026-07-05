import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ==========================================
# 1. Cấu hình Thiết bị (Device Setup)
# ==========================================
# Tự động chọn MPS cho Mac Silicon, CUDA cho Nvidia, hoặc CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Đang sử dụng thiết bị: {device}")

# ==========================================
# 2. Giả lập Dữ liệu (Thay bằng dữ liệu của bạn)
# ==========================================
# Ví dụ: Phân loại câu văn bản khoa học thành 2 lớp (0 hoặc 1)
texts = [
    "We use deep learning methods to classify citation intents in scientific papers.",
    "The sample was heated to 100 degrees Celsius and monitored for 24 hours.",
    "Previous studies failed to address the limitation of small-scale datasets.",
    "The patient showed significant improvement after the treatment protocol.",
    "Our proposed SynIntent dataset outperforms existing benchmarks significantly.",
    "We analyze the chemical composition of the soil samples collected from Mars."
] * 20  # Nhân bản lên để có đủ dữ liệu chạy demo

labels = [1, 0, 1, 0, 1, 0] * 20 

# Chia tập Train / Validation (80/20)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# ==========================================
# 3. Khởi tạo Tokenizer và Dataset
# ==========================================
# Sử dụng SciBERT (scivocab-uncased là bản phổ biến cho văn bản khoa học nói chung)
MODEL_NAME = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Khởi tạo DataLoader
BATCH_SIZE = 8
MAX_LEN = 128

train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ==========================================
# 4. Khởi tạo Mô hình & Optimizer
# ==========================================
# num_labels=2 cho bài toán phân loại nhị phân
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model = model.to(device)

EPOCHS = 3
LEARNING_RATE = 2e-5

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
total_steps = len(train_loader) * EPOCHS

# Hàm tối ưu giảm dần Learning Rate (Scheduler)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# ==========================================
# 5. Vòng lặp Huấn luyện (Training Loop)
# ==========================================
def train_epoch(model, data_loader, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    
    for batch in data_loader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        # Clip gradient để tránh bùng nổ gradient (gradient exploding)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
    return total_loss / len(data_loader)

# ==========================================
# 6. Hàm Đánh giá (Evaluation)
# ==========================================
def eval_model(model, data_loader, device):
    model.eval()
    predictions = []
    real_values = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            
            predictions.extend(preds)
            real_values.extend(labels.cpu().numpy())
            
    return accuracy_score(real_values, predictions), classification_report(real_values, predictions, zero_division=0)

# ==========================================
# 7. Tiến hành Chạy Huấn luyện
# ==========================================
print("Bắt đầu huấn luyện...")
for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch + 1}/{EPOCHS} ---")
    train_loss = train_epoch(model, train_loader, optimizer, device, scheduler)
    print(f"Train Loss: {train_loss:.4f}")
    
    val_acc, val_report = eval_model(model, val_loader, device)
    print(f"Validation Accuracy: {val_acc:.4f}")
    print("Classification Report:\n", val_report)

# ==========================================
# 8. Lưu Mô hình (Save Model)
# ==========================================
output_dir = "./scibert_binary_model"
os.makedirs(output_dir, exist_ok=True)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"\nHuấn luyện hoàn tất! Mô hình đã được lưu tại: {output_dir}")