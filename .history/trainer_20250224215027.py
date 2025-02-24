import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import List, Callable, Union
from functools import partial

from binary import BinaryHead

class MultiModelBinaryTrainer:
    def __init__(
        self,
        models: List[nn.Module],
        tokenizers: List[any],
        embedding_funcs: List[Callable],
        device: torch.device,
        temp: float = 1.0,
        num_trainable_layers: int = 2
    ):
        self.models = models  # 用于直接获取本地embedding的模型
        self.tokenizers = tokenizers
        self.embedding_funcs = embedding_funcs  # 用于API方式获取embedding的函数
        self.device = device
        self.binary_head = BinaryHead(temp=temp).to(device)
        self.criterion = nn.CosineEmbeddingLoss()
        
        for model in self.models:
            self._prepare_model_for_training(model, num_trainable_layers)
            
        params = list(self.binary_head.parameters())
        if num_trainable_layers > 0:
            for model in self.models:
                params.extend(filter(lambda p: p.requires_grad, model.parameters()))
        self.optimizer = optim.Adam(params, lr=3e-5)

    def _prepare_model_for_training(self, model, num_trainable_layers):
        """准备模型训练，冻结或解冻相应层"""
        for param in model.parameters():
            param.requires_grad = False
        
        if num_trainable_layers > 0:
            if hasattr(model, 'encoder'):
                encoder_layers = model.encoder.layer
            elif hasattr(model, 'layers'):
                encoder_layers = model.layers
            else:
                raise ValueError("Model architecture not supported for partial training")
            
            for layer in encoder_layers[-num_trainable_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def get_embeddings(self, texts, model_idx):
        """统一获取embedding的接口"""
        if model_idx < len(self.models):  # 使用本地模型
            model = self.models[model_idx]
            tokenizer = self.tokenizers[model_idx]
            
            enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = model(**enc)
                emb = torch.nn.functional.normalize(out[0][:, 0], p=2, dim=1)
            return emb
        else:  # 使用API方式获取embedding
            func_idx = model_idx - len(self.models)
            embeddings = self.embedding_funcs[func_idx](prompts=texts)
            return torch.tensor(embeddings).to(self.device)

    def train_epoch(self, data, batch_size, current_iter, total_iters):
        for model in self.models:
            if any(p.requires_grad for p in model.parameters()):
                model.train()
            else:
                model.eval()
        self.binary_head.train()
        
        total_loss = 0.0
        num_batches = max(int(len(data) / batch_size + 0.99), 1)
        data_shuffled = data.sample(frac=1, random_state=42)

        for i in tqdm(range(num_batches)):
            current_temp = max(0.01, 1.0 - (current_iter + i) / total_iters)
            self.binary_head.temp = current_temp
            batch = data_shuffled[i * batch_size : (i + 1) * batch_size]
            batch_loss = 0
            
            # 对每个模型/函数计算embeddings并累积loss
            for idx in range(len(self.models) + len(self.embedding_funcs)):
                emb1 = self.get_embeddings(list(batch["sample1"]), idx)
                emb2 = self.get_embeddings(list(batch["sample2"]), idx)
                labels = torch.tensor(batch["pos"].values).float().to(self.device)

                bin_out1 = self.binary_head(emb1)
                bin_out2 = self.binary_head(emb2)
                batch_loss += self.criterion(bin_out1, bin_out2, labels)

            # 平均所有模型的loss
            loss = batch_loss / (len(self.models) + len(self.embedding_funcs))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()

        return total_loss / len(data)

    def eval_epoch(self, data, batch_size):
        self.binary_head.eval()
        for model in self.models:
            model.eval()
            
        total_loss = 0.0
        num_batches = max(int(len(data) / batch_size + 0.99), 1)

        with torch.no_grad():
            for i in tqdm(range(num_batches), desc="Testing"):
                batch = data[i * batch_size : (i + 1) * batch_size]
                batch_loss = 0
                
                for idx in range(len(self.models) + len(self.embedding_funcs)):
                    emb1 = self.get_embeddings(list(batch["sample1"]), idx)
                    emb2 = self.get_embeddings(list(batch["sample2"]), idx)
                    labels = torch.tensor(batch["pos"].values).float().to(self.device)

                    out1 = self.binary_head(emb1)
                    out2 = self.binary_head(emb2)
                    batch_loss += self.criterion(out1, out2, labels)

                total_loss += batch_loss.item() / (len(self.models) + len(self.embedding_funcs))

        return total_loss / len(data)


    def train_binary_head(self, train_data, test_data, epochs, batch_size, output_dir="project/models/binary_head"):
        num_batches_per_epoch = max(int(len(train_data) / batch_size + 0.99), 1)
        total_iters = epochs * num_batches_per_epoch
        
        current_iter = 0
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_data, batch_size, current_iter, total_iters)
            current_iter += num_batches_per_epoch
            print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Current Temp: {self.binary_head.temp:.4f}")

        # 保存训练好的模型
        print("Saving model...")
        self.binary_head.save_model(output_dir)
        
        # 重新加载模型进行评估
        print("Reloading model for evaluation...")
        self.binary_head = BinaryHead.load_model(
            f"{output_dir}/binary_head_full.pt",
            self.device
        )
        
        test_loss = self.eval_epoch(test_data, batch_size)
        print(f"Test Loss: {test_loss:.4f}")

        return self.binary_head


def train(
    models: List[nn.Module],
    tokenizers: List[any],
    embedding_funcs: List[Callable],
    train_data,
    test_data,
    device: torch.device,
    epochs: int = 10,
    lr: float = 2e-5,
    batch_size: int = 4,
    temp: float = 1.0,
    num_trainable_layers: int = 1,
    output_dir: str = "project/models/binary_head"
) -> BinaryHead:
    trainer = MultiModelBinaryTrainer(
        models=models,
        tokenizers=tokenizers,
        embedding_funcs=embedding_funcs,
        device=device,
        temp=temp,
        num_trainable_layers=num_trainable_layers
    )
    
    return trainer.train_binary_head(
        train_data=train_data,
        test_data=test_data,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=output_dir
    )