import os
import csv
import re
import statistics
import yaml

ITERATION_COUNT = 5

def find_versions(logs_path):
    """返回 logs 目录下所有 version_x 文件夹的绝对路径"""
    versions = []
    for d in os.listdir(logs_path):
        if d.startswith('version_') and os.path.isdir(os.path.join(logs_path, d)):
            versions.append(os.path.join(logs_path, d))
    return versions

def get_hparams(hparams_path):
    try:
        with open(hparams_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def normalize_field(value):
    if value is None:
        return ''
    if isinstance(value, list):
        return '+'.join(str(v) for v in value)
    return str(value)

def get_last_test_acc(metrics_path):
    """读取 metrics.csv，返回最后一行的 test_acc(float)"""
    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            for row in reversed(rows):
                value = row.get('test_acc')
                if value not in (None, ''):
                    return float(value)
        return None
    except Exception:
        return None

def find_latest_version(logs_path):
    versions = find_versions(logs_path)
    if not versions:
        return None
    def version_key(path):
        match = re.search(r'version_(\\d+)', path)
        return int(match.group(1)) if match else -1
    return max(versions, key=version_key)

def extract_iteration(exp_name):
    match = re.search(r'_it(\\d+)', exp_name)
    if match:
        return int(match.group(1))
    return None

def main():
    save_root = 'F:\\eaff\\figure\\save'
    raw_rows = []
    for task in os.listdir(save_root):
        task_path = os.path.join(save_root, task)
        if not os.path.isdir(task_path):
            continue
        for model in os.listdir(task_path):
            model_path = os.path.join(task_path, model)
            if not os.path.isdir(model_path):
                continue
            for exp in os.listdir(model_path):
                exp_path = os.path.join(model_path, exp)
                logs_path = os.path.join(exp_path, 'logs')
                if not os.path.isdir(logs_path):
                    continue
                version_path = find_latest_version(logs_path)
                if not version_path:
                    continue
                hparams_path = os.path.join(version_path, 'hparams.yaml')
                metrics_path = os.path.join(version_path, 'metrics.csv')
                hparams = get_hparams(hparams_path)
                test_acc = get_last_test_acc(metrics_path)
                raw_rows.append({
                    'dataset_task': hparams.get('dataset_task', ''),
                    'model': hparams.get('model', ''),
                    'exp': exp,
                    'iteration': extract_iteration(exp),
                    'target': normalize_field(hparams.get('target')),
                    'source': normalize_field(hparams.get('source')),
                    'test_acc': test_acc,
                })

    out_raw = os.path.join(save_root, 'result_acc_raw.csv')
    with open(out_raw, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['dataset_task', 'model', 'exp', 'iteration', 'target', 'source', 'test_acc'],
        )
        writer.writeheader()
        writer.writerows(raw_rows)

    grouped = {}
    for row in raw_rows:
        key = (row['dataset_task'], row['model'], row['target'], row['source'])
        grouped.setdefault(key, {})
        iteration = row['iteration']
        if iteration is None:
            continue
        grouped[key][iteration] = row['test_acc']

    summary_rows = []
    for key, acc_map in grouped.items():
        dataset_task, model, target, source = key
        acc_values = [v for v in acc_map.values() if isinstance(v, float)]
        if acc_values:
            mean_acc = statistics.mean(acc_values)
            std_acc = statistics.stdev(acc_values) if len(acc_values) > 1 else 0.0
            best_acc = max(acc_values)
        else:
            mean_acc = ''
            std_acc = ''
            best_acc = ''
        row = {
            'dataset_task': dataset_task,
            'model': model,
            'target': target,
            'source': source,
            'mean_acc': mean_acc,
            'std_acc': std_acc,
            'best_acc': best_acc,
        }
        for i in range(ITERATION_COUNT):
            row[f'acc_it{i}'] = acc_map.get(i, '')
        summary_rows.append(row)

    summary_path = os.path.join(save_root, 'result_acc_summary.csv')
    fieldnames = [
        'dataset_task',
        'model',
        'target',
        'source',
    ] + [f'acc_it{i}' for i in range(ITERATION_COUNT)] + [
        'mean_acc',
        'std_acc',
        'best_acc',
    ]
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f'已保存到 {summary_path}，共 {len(summary_rows)} 条记录。')
    print(f'原始记录保存到 {out_raw}，共 {len(raw_rows)} 条记录。')

if __name__ == '__main__':
    main()
