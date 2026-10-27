try:
    import torch
    print('PyTorch:', torch.__version__)
    print('CUDA khả dụng:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
        print('Số GPU CUDA:', torch.cuda.device_count())
    else:
        print('YOLO sẽ chạy CPU khi chọn Tự động. Nếu máy có NVIDIA GPU, cần cài bản PyTorch hỗ trợ CUDA phù hợp.')
except Exception as e:
    print('Không kiểm tra được PyTorch:', e)
