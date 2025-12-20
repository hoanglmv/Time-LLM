# -*- coding: utf-8 -*-
# File này chứa các hàm để vẽ và trực quan hóa kết quả huấn luyện.

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_loss(train_loss_history, vali_loss_history, save_path):
    """
    Vẽ biểu đồ training loss và validation loss qua các epochs và lưu lại.
    
    Args:
        train_loss_history (list): Danh sách các giá trị training loss mỗi epoch.
        vali_loss_history (list): Danh sách các giá trị validation loss mỗi epoch.
        save_path (str): Đường dẫn để lưu file ảnh (ví dụ: 'figures/my_model_loss.png').
    """
    try:
        epochs = range(1, len(train_loss_history) + 1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_loss_history, 'b-o', label='Training Loss')
        plt.plot(epochs, vali_loss_history, 'r-o', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Đảm bảo thư mục lưu trữ tồn tại
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path)
        plt.close() # Đóng plot để giải phóng bộ nhớ
        print(f"✅ Đã lưu biểu đồ loss tại: {save_path}")

    except Exception as e:
        print(f"⚠️ Lỗi khi vẽ biểu đồ loss: {e}")

def plot_classification_metrics(metrics_history, save_path):
    """
    Vẽ biểu đồ các chỉ số classification (Accuracy, F1, Precision, Recall) 
    qua các epochs và lưu lại.
    
    Args:
        metrics_history (dict): Dictionary chứa list giá trị của mỗi metric. 
                                 Ví dụ: {'accuracy': [0.8, 0.85], 'f1': [0.7, 0.75], ...}
        save_path (str): Đường dẫn để lưu file ảnh (ví dụ: 'figures/my_model_metrics.png').
    """
    try:
        num_epochs = 0
        # Lấy số epochs từ metric đầu tiên có trong history
        for metric in metrics_history:
            if len(metrics_history[metric]) > 0:
                num_epochs = len(metrics_history[metric])
                break
        if num_epochs == 0:
            print("⚠️ Không có dữ liệu metrics để vẽ biểu đồ.")
            return

        epochs = range(1, num_epochs + 1)
        
        plt.figure(figsize=(12, 8))
        
        for metric_name, values in metrics_history.items():
            if values: # Chỉ vẽ nếu có dữ liệu
                plt.plot(epochs, values, '-o', label=metric_name.capitalize())
        
        plt.title('Classification Metrics over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1.05) # Giới hạn trục y từ 0 đến 1.05

        # Đảm bảo thư mục lưu trữ tồn tại
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Đã lưu biểu đồ metrics tại: {save_path}")

    except Exception as e:
        print(f"⚠️ Lỗi khi vẽ biểu đồ metrics: {e}")
