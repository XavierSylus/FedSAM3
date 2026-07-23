# 快速修复脚本：更新配置文件中的 client_id

import yaml
from pathlib import Path

configs = [
    'configs/exp_group_a.yaml',
    'configs/exp_group_b.yaml',
    'configs/exp_group_c.yaml'
]

# client_id 映射
id_mapping = {
    'client1': 'client_1',
    'client2': 'client_2',
    'client3': 'client_3'
}

for config_file in configs:
    config_path = Path(config_file)
    if not config_path.exists():
        print(f"跳过不存在的文件: {config_file}")
        continue

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 更新 client_id
    if 'federated' in config and 'clients' in config['federated']:
        for client in config['federated']['clients']:
            old_id = client.get('client_id', '')
            if old_id in id_mapping:
                new_id = id_mapping[old_id]
                client['client_id'] = new_id
                print(f"{config_file}: {old_id} -> {new_id}")

    # 保存
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"✅ 已更新: {config_file}\n")

print("所有配置文件已更新！现在 client_id 与目录名称匹配。")
