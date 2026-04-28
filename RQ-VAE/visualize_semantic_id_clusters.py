import argparse
import os

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.manifold import MDS
try:
    from adjustText import adjust_text
except ImportError:
    adjust_text = None


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def decode_value(value):
    if isinstance(value, (bytes, bytearray)):
        return value.decode('utf-8', errors='ignore')
    return str(value)


def longest_common_prefix_len(code_a, code_b):
    length = 0
    for a, b in zip(code_a, code_b):
        if a == b:
            length += 1
        else:
            break
    return length


def prefix_distance_matrix(codes_3d):
    n = len(codes_3d)
    distances = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            lcp = longest_common_prefix_len(codes_3d[i], codes_3d[j])
            dist = 3 - lcp
            distances[i, j] = dist
            distances[j, i] = dist
    return distances


def load_course_info(course_info_path):
    with h5py.File(course_info_path, 'r') as f:
        item_ids = [decode_value(x) for x in f['item_id'][:]]
        item_names = [decode_value(x) for x in f['item_name'][:]]
        item_infos = [decode_value(x) for x in f['item_info'][:]]
    return pd.DataFrame(
        {
            'course_index': np.arange(1, len(item_ids) + 1, dtype=int),
            'item_id': item_ids,
            'item_name': item_names,
            'item_info': item_infos,
        }
    )


def load_aligned_codes(codes_path, expected_count):
    codes = np.load(codes_path, allow_pickle=True)
    if len(codes) < expected_count + 1:
        raise ValueError(
            f'语义ID数量不足，期望至少 {expected_count + 1} 条（包含0号padding），实际为 {len(codes)}'
        )
    aligned = codes[1: expected_count + 1]
    if aligned.shape[0] != expected_count:
        raise ValueError(f'对齐后语义ID数量错误，期望 {expected_count}，实际为 {aligned.shape[0]}')
    return aligned


def build_dataframe(course_df, aligned_codes):
    code_cols = ['code_1', 'code_2', 'code_3', 'code_4']
    code_df = pd.DataFrame(aligned_codes, columns=code_cols)
    df = pd.concat([course_df.reset_index(drop=True), code_df], axis=1)
    df['prefix1'] = df['code_1'].astype(str)
    df['prefix2'] = df['code_1'].astype(str) + '-' + df['code_2'].astype(str)
    df['prefix3'] = (
        df['code_1'].astype(str)
        + '-'
        + df['code_2'].astype(str)
        + '-'
        + df['code_3'].astype(str)
    )
    return df


def compute_prefix_stats(df):
    prefix1_stats = df['prefix1'].value_counts().sort_index().rename_axis('prefix1').reset_index(name='course_count')
    prefix2_stats = df['prefix2'].value_counts().rename_axis('prefix2').reset_index(name='course_count')
    prefix2_stats = prefix2_stats.sort_values(['course_count', 'prefix2'], ascending=[False, True])
    return prefix1_stats, prefix2_stats


def plot_prefix1_counts(prefix1_stats, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(prefix1_stats['prefix1'].astype(str), prefix1_stats['course_count'], color='#4C78A8')
    ax.set_title('一级语义前缀课程数量分布')
    ax.set_xlabel('一级前缀')
    ax.set_ylabel('课程数量')
    for i, v in enumerate(prefix1_stats['course_count']):
        ax.text(i, v + 1, str(v), ha='center', va='bottom', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'prefix1_counts.png'), dpi=220)
    plt.close(fig)


def plot_prefix2_top_counts(prefix2_stats, output_dir, top_k=20):
    top_df = prefix2_stats.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_df['prefix2'], top_df['course_count'], color='#F58518')
    ax.set_title(f'课程数量最多的前{top_k}个二级语义前缀')
    ax.set_xlabel('课程数量')
    ax.set_ylabel('二级前缀')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'prefix2_top_{top_k}_counts.png'), dpi=220)
    plt.close(fig)


def plot_prefix_tree(df, output_dir, focus_prefix1='0', top_prefix2=4, top_prefix3=4):
    sub1 = df[df['prefix1'] == str(focus_prefix1)].copy()
    if sub1.empty:
        raise ValueError(f'未找到 prefix1={focus_prefix1} 的课程数据')

    prefix1_count = int(len(sub1))
    prefix2_counts = sub1.groupby('prefix2').size().sort_values(ascending=False).head(top_prefix2)

    tree_rows = []
    layout_rows = []
    for prefix2, prefix2_count in prefix2_counts.items():
        sub2 = sub1[sub1['prefix2'] == prefix2].copy()
        prefix3_counts = sub2.groupby('prefix3').size().sort_values(ascending=False).head(top_prefix3)
        for prefix3, prefix3_count in prefix3_counts.items():
            sub3 = sub2[sub2['prefix3'] == prefix3].copy()
            examples = ' | '.join(sub3['item_name'].head(2).tolist())
            row = {
                'prefix1': str(focus_prefix1),
                'prefix1_count': prefix1_count,
                'prefix2': prefix2,
                'prefix2_count': int(prefix2_count),
                'prefix3': prefix3,
                'prefix3_count': int(prefix3_count),
                'example_courses': examples,
            }
            tree_rows.append(row)
            layout_rows.append(row)

    tree_df = pd.DataFrame(tree_rows)

    # SCI palette - unified by level
    color_prefix1 = '#4C72B0'  # blue
    color_prefix2 = '#55A868'  # green
    color_prefix3 = '#DD8452'  # orange
    sci_gray = '#6C757D'

    total_rows = max(1, len(layout_rows))
    fig_height = max(7.0, total_rows * 0.54)
    fig, ax = plt.subplots(figsize=(11.2, fig_height))
    ax.axis('off')

    x1, x2, x3 = 0.08, 0.28, 0.52
    y_positions = np.linspace(0.90, 0.10, total_rows)
    prefix2_centers = {}
    for i, row in enumerate(layout_rows):
        row['_y'] = float(y_positions[i])
        prefix2_centers.setdefault(row['prefix2'], []).append(row['_y'])
    prefix2_centers = {k: float(np.mean(v)) for k, v in prefix2_centers.items()}

    y1 = 0.50
    ax.text(
        x1,
        y1,
        f'prefix1={focus_prefix1}\nCourses={prefix1_count}',
        ha='center',
        va='center',
        fontsize=12,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.36', facecolor=color_prefix1, alpha=0.16, edgecolor=color_prefix1, linewidth=1.4)
    )

    drawn_prefix2 = set()
    for idx, row in enumerate(layout_rows):
        prefix2 = row['prefix2']
        prefix3 = row['prefix3']
        y2 = prefix2_centers[prefix2]
        y3 = row['_y']

        if prefix2 not in drawn_prefix2:
            ax.plot([x1 + 0.04, x2 - 0.05], [y1, y2], color=sci_gray, alpha=0.65, linewidth=1.1)
            ax.text(
                x2,
                y2,
                f'{prefix2}\nCourses={row["prefix2_count"]}',
                ha='center',
                va='center',
                fontsize=9,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.28', facecolor=color_prefix2, alpha=0.12, edgecolor=color_prefix2, linewidth=1.1)
            )
            drawn_prefix2.add(prefix2)

        short_example = row["example_courses"][:18]
        ax.plot([x2 + 0.05, x3 - 0.05], [y2, y3], color=color_prefix2, alpha=0.65, linewidth=0.95)
        ax.text(
            x3,
            y3,
            f'{prefix3}\nCourses={row["prefix3_count"]}\n{short_example}',
            ha='center',
            va='center',
            fontsize=7.6,
            bbox=dict(boxstyle='round,pad=0.20', facecolor='white', alpha=0.98, edgecolor=color_prefix3, linewidth=0.95)
        )

    legend_handles = [
        Line2D([0], [0], marker='s', color='w', label='Level 1: prefix1', markerfacecolor=color_prefix1, markeredgecolor=color_prefix1, markersize=10, alpha=0.7),
        Line2D([0], [0], marker='s', color='w', label='Level 2: prefix2', markerfacecolor=color_prefix2, markeredgecolor=color_prefix2, markersize=9, alpha=0.5),
        Line2D([0], [0], marker='s', color='w', label='Level 3: prefix3', markerfacecolor=color_prefix3, markeredgecolor=color_prefix3, markersize=8, alpha=0.4),
    ]
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.01, 0.99), frameon=False, fontsize=8.5)

    ax.set_xlim(0.0, 0.78)
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout(rect=[0.01, 0.01, 0.95, 0.98])
    fig.savefig(os.path.join(output_dir, 'semantic_id_prefix1_0_tree.png'), dpi=300)
    plt.close(fig)

    tree_df.to_csv(os.path.join(output_dir, 'semantic_id_prefix1_0_tree_summary.csv'), index=False, encoding='utf-8-sig')


def plot_prefix_mds(df, output_dir, annotate_per_prefix=2):
    codes_3d = df[['code_1', 'code_2', 'code_3']].to_numpy(dtype=int)
    distances = prefix_distance_matrix(codes_3d)
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress='auto', n_init=4)
    coords = mds.fit_transform(distances)
    plot_df = df.copy()
    plot_df['x'] = coords[:, 0]
    plot_df['y'] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(11, 8))
    prefixes = sorted(plot_df['prefix1'].unique(), key=lambda x: int(x))
    cmap = matplotlib.colormaps.get_cmap('tab10')
    texts = []
    for idx, prefix in enumerate(prefixes):
        sub = plot_df[plot_df['prefix1'] == prefix]
        ax.scatter(sub['x'], sub['y'], s=40, alpha=0.8, label=f'prefix1={prefix}', color=cmap(idx / max(1, len(prefixes) - 1)))
        center_x = sub['x'].median()
        center_y = sub['y'].median()
        sub = sub.assign(center_dist=(sub['x'] - center_x) ** 2 + (sub['y'] - center_y) ** 2)
        to_annotate = sub.sort_values(['center_dist', 'course_index']).head(annotate_per_prefix)
        for _, row in to_annotate.iterrows():
            texts.append(ax.text(row['x'], row['y'], row['item_name'][:12], fontsize=8, alpha=0.9))

    if adjust_text is not None and texts:
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5, lw=0.8),
            only_move={'points': 'y', 'text': 'xy'}
        )

    ax.set_title('基于前三层语义ID前缀距离的课程聚类图')
    ax.set_xlabel('MDS-1')
    ax.set_ylabel('MDS-2')
    ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'semantic_id_prefix_mds.png'), dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--codes', default='../data/course/course_rqvae_codes.npy')
    parser.add_argument('--course-info', default='../data/course_info.h5')
    parser.add_argument('--output-dir', default='./semantic_id_viz')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    course_df = load_course_info(args.course_info)
    aligned_codes = load_aligned_codes(args.codes, expected_count=len(course_df))
    df = build_dataframe(course_df, aligned_codes)

    df.to_csv(os.path.join(args.output_dir, 'course_semantic_id_alignment.csv'), index=False, encoding='utf-8-sig')

    prefix1_stats, prefix2_stats = compute_prefix_stats(df)

    plot_prefix1_counts(prefix1_stats, args.output_dir)
    plot_prefix2_top_counts(prefix2_stats, args.output_dir, top_k=20)
    plot_prefix_tree(df, args.output_dir)
    plot_prefix_mds(df, args.output_dir)

    print(f'已完成可视化，结果保存在: {os.path.abspath(args.output_dir)}')
    print('对齐规则: 使用 course_index=1~706 对应 codes[1:707]，跳过 codes[0] 作为 padding')
    if adjust_text is None:
        print('未检测到 adjustText，已使用少量居中标签但未进行自动避让。可执行: pip install adjustText')


if __name__ == '__main__':
    main()
