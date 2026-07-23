"""
extract_main_table.py — 从 A/B/C 三组 training_history.json 自动提取论文主表

用法：
    python scripts/extract_main_table.py --data_dir server_data

输出：
    1. 终端打印 Markdown 主表
    2. server_data/main_table.csv
    3. server_data/training_curves.png（训练曲线对比图）
    4. server_data/grad_conflict.png（梯度冲突角曲线）
"""

import json
import csv
import sys
import math
import argparse
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_history(json_path: Path) -> dict:
    with open(json_path, 'r', encoding='utf-8') as f:
        content = f.read()
        content = content.replace('Infinity', '999999.0')
        content = content.replace('NaN', 'null')
        return json.loads(content)


def extract_metrics(history: dict) -> dict:
    val_metrics = history.get('val_metrics', [])
    rounds = history.get('rounds', [])
    grad_conflicts = history.get('grad_conflict_deg', [])

    best_dice = 0.0
    best_round = 0
    best_hd95 = 999999.0
    final_dice = 0.0
    final_hd95 = 999999.0
    final_round = max(rounds) if rounds else 0

    for vm in val_metrics:
        d = vm.get('dice', 0.0)
        if d is None:
            d = 0.0
        h = vm.get('hd95', 999999.0)
        if h is None or h >= 999999.0:
            h = 999999.0
        r = vm.get('round', 0)

        if d > best_dice:
            best_dice = d
            best_round = r
            best_hd95 = h

    if val_metrics:
        last = val_metrics[-1]
        final_dice = last.get('dice', 0.0) or 0.0
        final_hd95 = last.get('hd95', 999999.0)
        if final_hd95 is None or final_hd95 >= 999999.0:
            final_hd95 = 999999.0

    valid_conflicts = [c for c in grad_conflicts if c is not None]
    avg_conflict = sum(valid_conflicts) / len(valid_conflicts) if valid_conflicts else None

    final_stats = history.get('final_stats', {})
    run_meta = history.get('run_metadata', {})

    return {
        'total_rounds': final_round,
        'best_dice': best_dice,
        'best_round': best_round,
        'best_hd95': best_hd95,
        'final_dice': final_dice,
        'final_hd95': final_hd95,
        'avg_grad_conflict': avg_conflict,
        'total_params': final_stats.get('total_params', 'N/A'),
        'trainable_params': final_stats.get('trainable_params', 'N/A'),
        'seed': run_meta.get('seed', 'N/A'),
        'git_commit': run_meta.get('git_commit', 'N/A'),
        'val_metrics_list': val_metrics,
        'grad_conflict_list': grad_conflicts,
        'rounds_list': rounds,
        'avg_losses': history.get('avg_losses', []),
        'avg_seg_losses': history.get('avg_seg_losses', []),
        'avg_cream_losses': history.get('avg_cream_losses', []),
    }


def format_hd95(val: float) -> str:
    if val >= 999999.0:
        return '∞'
    return f'{val:.2f}'


def print_main_table(results: dict):
    labels = {
        'group_a': ('A', 'Image-only baseline'),
        'group_b': ('B', 'Multimodal + FedAvg (text in agg)'),
        'group_c': ('C', 'Decoupled distillation (text-free agg)'),
    }

    print('\n' + '=' * 100)
    print('                         论文主表 (Table 1: A/B/C 对比)')
    print('=' * 100)

    header = f'| {"Group":^6} | {"Setting":^45} | {"Best Dice":^10} | {"Best Rd":^8} | {"Final Dice":^11} | {"Final HD95":^11} | {"GradConflict":^13} |'
    sep = f'|{"-"*8}|{"-"*47}|{"-"*12}|{"-"*10}|{"-"*13}|{"-"*13}|{"-"*15}|'

    print(header)
    print(sep)

    for group_key in ['group_a', 'group_b', 'group_c']:
        if group_key not in results:
            continue
        m = results[group_key]
        label, setting = labels[group_key]
        gc = f"{m['avg_grad_conflict']:.1f}°" if m['avg_grad_conflict'] is not None else 'N/A'
        row = f"| {label:^6} | {setting:<45} | {m['best_dice']:^10.4f} | {m['best_round']:^8} | {m['final_dice']:^11.4f} | {format_hd95(m['final_hd95']):^11} | {gc:^13} |"
        print(row)

    print(sep)
    print()

    # 元数据
    print('实验元数据:')
    for group_key in ['group_a', 'group_b', 'group_c']:
        if group_key not in results:
            continue
        m = results[group_key]
        label, _ = labels[group_key]
        print(f'  Group {label}: rounds={m["total_rounds"]}, params={m["total_params"]}, trainable={m["trainable_params"]}, seed={m["seed"]}')

    # 差异分析
    if 'group_b' in results and 'group_c' in results:
        b = results['group_b']
        c = results['group_c']
        dice_diff = c['best_dice'] - b['best_dice']
        print(f'\n  C-B Dice差异: {dice_diff:+.4f} ({"C 优于 B ✓" if dice_diff > 0 else "B 优于 C ✗"})')
        if b['avg_grad_conflict'] is not None and c['avg_grad_conflict'] is not None:
            gc_diff = b['avg_grad_conflict'] - c['avg_grad_conflict']
            print(f'  B-C 梯度冲突角差异: {gc_diff:+.1f}° ({"B冲突更大 ✓" if gc_diff > 0 else "C冲突更大 ✗"})')


def save_csv(results: dict, output_path: Path):
    rows = []
    for group_key in ['group_a', 'group_b', 'group_c']:
        if group_key not in results:
            continue
        m = results[group_key]
        rows.append({
            'group': group_key,
            'best_dice': m['best_dice'],
            'best_round': m['best_round'],
            'best_hd95': m['best_hd95'],
            'final_dice': m['final_dice'],
            'final_hd95': m['final_hd95'],
            'avg_grad_conflict': m['avg_grad_conflict'],
            'total_rounds': m['total_rounds'],
            'total_params': m['total_params'],
            'trainable_params': m['trainable_params'],
        })

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'\n[OK] CSV 已保存: {output_path}')


def plot_training_curves(results: dict, output_dir: Path):
    if not HAS_MPL:
        print('[SKIP] matplotlib 未安装，跳过绘图')
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('A/B/C Training Curves Comparison', fontsize=14, fontweight='bold')

    colors = {'group_a': '#2196F3', 'group_b': '#F44336', 'group_c': '#4CAF50'}
    labels = {'group_a': 'Group A (Image-only)', 'group_b': 'Group B (Text in Agg)', 'group_c': 'Group C (Decoupled)'}

    # 1: Dice 曲线
    ax = axes[0, 0]
    for gk in ['group_a', 'group_b', 'group_c']:
        if gk not in results:
            continue
        vm = results[gk]['val_metrics_list']
        if vm:
            rounds = [v.get('round', i+1) for i, v in enumerate(vm)]
            dices = [v.get('dice', 0) or 0 for v in vm]
            ax.plot(rounds, dices, color=colors[gk], label=labels[gk], linewidth=1.5)
    ax.set_xlabel('Round')
    ax.set_ylabel('Dice')
    ax.set_title('Validation Dice')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2: Loss 曲线
    ax = axes[0, 1]
    for gk in ['group_a', 'group_b', 'group_c']:
        if gk not in results:
            continue
        losses = results[gk]['avg_losses']
        if losses:
            ax.plot(range(1, len(losses)+1), losses, color=colors[gk], label=labels[gk], linewidth=1.5)
    ax.set_xlabel('Round')
    ax.set_ylabel('Avg Loss')
    ax.set_title('Training Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3: Cream Loss 曲线
    ax = axes[1, 0]
    for gk in ['group_b', 'group_c']:
        if gk not in results:
            continue
        cream = results[gk]['avg_cream_losses']
        if cream:
            ax.plot(range(1, len(cream)+1), cream, color=colors[gk], label=labels[gk], linewidth=1.5)
    ax.set_xlabel('Round')
    ax.set_ylabel('Cream Loss')
    ax.set_title('Cream (Contrastive) Loss')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4: 梯度冲突角曲线
    ax = axes[1, 1]
    for gk in ['group_b', 'group_c']:
        if gk not in results:
            continue
        gc = results[gk]['grad_conflict_list']
        if gc:
            valid_rounds = []
            valid_gc = []
            for i, v in enumerate(gc):
                if v is not None:
                    valid_rounds.append(i + 1)
                    valid_gc.append(v)
            if valid_gc:
                ax.plot(valid_rounds, valid_gc, color=colors[gk], label=labels[gk], linewidth=1.5)
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90° (orthogonal)')
    ax.set_xlabel('Round')
    ax.set_ylabel('Grad Conflict Angle (°)')
    ax.set_title('Adapter Gradient Conflict')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / 'training_curves.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[OK] 训练曲线图已保存: {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Extract A/B/C main table from training history')
    parser.add_argument('--data_dir', type=str, default='server_data',
                        help='Directory containing group_a/b/c subdirectories with training_history.json')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f'[ERROR] 数据目录不存在: {data_dir}')
        print(f'请先下载服务器数据到 {data_dir}/ 目录')
        sys.exit(1)

    results = {}
    for group in ['group_a', 'group_b', 'group_c']:
        json_path = data_dir / group / 'training_history.json'
        if not json_path.exists():
            print(f'[WARN] 未找到 {json_path}，跳过 {group}')
            continue

        print(f'[Loading] {json_path}')
        history = load_history(json_path)
        results[group] = extract_metrics(history)
        print(f'  → {group}: {results[group]["total_rounds"]} rounds, best_dice={results[group]["best_dice"]:.4f}')

    if not results:
        print('[ERROR] 未加载任何数据，请检查文件路径')
        sys.exit(1)

    print_main_table(results)
    save_csv(results, data_dir / 'main_table.csv')
    plot_training_curves(results, data_dir)


if __name__ == '__main__':
    main()
