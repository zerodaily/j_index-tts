#!/usr/bin/env python3
"""测试 IndexTTS2 模型加载和显存占用"""
import os
import sys
import torch

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, "indextts"))

def get_gpu_memory():
    """获取当前GPU显存占用"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        return allocated, reserved
    return 0, 0

def test_load_model():
    print("=" * 50)
    print("IndexTTS2 模型加载测试")
    print("=" * 50)
    print(f"GPU可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"总显存: {props.total_memory / 1024**3:.2f} GB")

    print("\n[1/4] 加载 IndexTTS2 (FP16 模式)...")
    allocated_before, _ = get_gpu_memory()
    print(f"  加载前显存占用: {allocated_before:.2f} GB")

    from indextts.infer_v2 import IndexTTS2

    tts = IndexTTS2(
        cfg_path="checkpoints/config.yaml",
        model_dir="checkpoints",
        use_fp16=True,        # 启用 FP16 节省显存
        use_cuda_kernel=False, # 禁用 CUDA kernel 节省显存
        use_deepspeed=False    # 禁用 DeepSpeed 节省显存
    )

    allocated_after, reserved = get_gpu_memory()
    print(f"  加载后显存占用: {allocated_after:.2f} GB")
    print(f"  显存预留: {reserved:.2f} GB")
    print("  ✓ 模型加载成功!")

    # 显存清理测试
    print("\n[2/4] 测试显存清理...")
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    allocated_after_clean, reserved_clean = get_gpu_memory()
    print(f"  清理后显存占用: {allocated_after_clean:.2f} GB")
    print(f"  显存预留: {reserved_clean:.2f} GB")

    # 简单推理测试
    print("\n[3/4] 测试推理 (短文本)...")
    test_text = "你好，欢迎使用 IndexTTS2。"
    test_audio = "examples/voice_01.wav"

    if os.path.exists(test_audio):
        try:
            tts.gr_progress = None  # 禁用进度回调
            output = tts.infer(
                spk_audio_prompt=test_audio,
                text=test_text,
                output_path="test_output.wav",
                verbose=False
            )
            allocated_infer, _ = get_gpu_memory()
            print(f"  推理后显存占用: {allocated_infer:.2f} GB")
            if output:
                print(f"  ✓ 推理成功! 输出: {output}")
            else:
                print(f"  ⚠ 推理未返回有效输出")
        except Exception as e:
            print(f"  ✗ 推理失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"  ⚠ 跳过推理测试: {test_audio} 不存在")

    print("\n[4/4] 显存状态汇总")
    print("=" * 50)
    allocated_final, reserved_final = get_gpu_memory()
    print(f"最终显存占用: {allocated_final:.2f} GB")
    print(f"最终显存预留: {reserved_final:.2f} GB")
    print(f"可用显存: {(props.total_memory / 1024**3) - allocated_final:.2f} GB")
    print("=" * 50)

    # 检查是否在22GB以内
    if allocated_final < 20:  # 留2GB余量
        print("✓ 显存使用优化成功! 模型可在22GB显存下运行。")
    else:
        print("⚠ 显存占用较高，可能需要进一步优化。")

    return tts

if __name__ == "__main__":
    os.chdir("/home/zerodaily/projects/index-tts")
    test_load_model()
