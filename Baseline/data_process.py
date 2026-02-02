import sqlite3
import pandas as pd
import h5py
import numpy as np
import os
import json
from collections import defaultdict

def extract_data_from_db(db_path, output_h5_path):
    
    conn = sqlite3.connect(db_path)
    
    # 提取课程信息 (class_index表)
    query_classes = """
    SELECT class_id, class_name, keywords_pos, keywords_neg, content, url
    FROM class_index
    ORDER BY class_id
    """
    df_classes = pd.read_sql_query(query_classes, conn)
    df_classes = df_classes.dropna(subset=['class_name'])
    
    # 提取用户交互记录 (interaction_records表)
    query_interactions = """
    SELECT id, student_id, class_id, class_name, keywords_pos, keywords_neg, preference
    FROM interaction_records
    ORDER BY student_id, id
    """
    df_interactions = pd.read_sql_query(query_interactions, conn)
    
    # 提取用户资料 (students表)
    query_students = """
    SELECT student_id, major, interest_long_profile
    FROM students
    """
    df_students = pd.read_sql_query(query_students, conn)
    conn.close()
    
    # 保存到HDF5文件  
    with h5py.File(output_h5_path, 'w') as hf:
        # 保存课程信息
        classes_group = hf.create_group('classes')
        
        # 保存课程ID列表
        class_ids = df_classes['class_id'].values
        classes_group.create_dataset('class_ids', data=class_ids, compression='gzip')
        
        # 保存课程名称 (转换为字节数组以支持压缩)
        class_names = df_classes['class_name'].fillna('').values.astype(str)
        class_names_bytes = np.array([s.encode('utf-8') for s in class_names], dtype=object)
        classes_group.create_dataset('class_names', data=class_names_bytes, compression='gzip')
        
        # 保存关键词 (正面和负面)
        keywords_pos = df_classes['keywords_pos'].fillna('').values.astype(str)
        keywords_neg = df_classes['keywords_neg'].fillna('').values.astype(str)
        keywords_pos_bytes = np.array([s.encode('utf-8') for s in keywords_pos], dtype=object)
        keywords_neg_bytes = np.array([s.encode('utf-8') for s in keywords_neg], dtype=object)
        classes_group.create_dataset('keywords_pos', data=keywords_pos_bytes, compression='gzip')
        classes_group.create_dataset('keywords_neg', data=keywords_neg_bytes, compression='gzip')
        
        # 保存课程内容和URL
        content = df_classes['content'].fillna('').values.astype(str)
        url = df_classes['url'].fillna('').values.astype(str)
        content_bytes = np.array([s.encode('utf-8') for s in content], dtype=object)
        url_bytes = np.array([s.encode('utf-8') for s in url], dtype=object)
        classes_group.create_dataset('content', data=content_bytes, compression='gzip')
        classes_group.create_dataset('url', data=url_bytes, compression='gzip')
        # 保存用户交互记录
        interactions_group = hf.create_group('interactions')
        interactions_group.create_dataset('record_ids', data=df_interactions['id'].values, compression='gzip')
        interactions_group.create_dataset('student_ids', data=df_interactions['student_id'].values, compression='gzip')
        interactions_group.create_dataset('class_ids', data=df_interactions['class_id'].values, compression='gzip')
        
        interaction_class_names = df_interactions['class_name'].fillna('').values.astype(str)
        interaction_keywords_pos = df_interactions['keywords_pos'].fillna('').values.astype(str)
        interaction_keywords_neg = df_interactions['keywords_neg'].fillna('').values.astype(str)
        interaction_preference = df_interactions['preference'].fillna('').values.astype(str)
        
        interaction_class_names_bytes = np.array([s.encode('utf-8') for s in interaction_class_names], dtype=object)
        interaction_keywords_pos_bytes = np.array([s.encode('utf-8') for s in interaction_keywords_pos], dtype=object)
        interaction_keywords_neg_bytes = np.array([s.encode('utf-8') for s in interaction_keywords_neg], dtype=object)
        interaction_preference_bytes = np.array([s.encode('utf-8') for s in interaction_preference], dtype=object)
        
        interactions_group.create_dataset('class_names', data=interaction_class_names_bytes, compression='gzip')
        interactions_group.create_dataset('keywords_pos', data=interaction_keywords_pos_bytes, compression='gzip')
        interactions_group.create_dataset('keywords_neg', data=interaction_keywords_neg_bytes, compression='gzip')
        interactions_group.create_dataset('preference', data=interaction_preference_bytes, compression='gzip')
        
        # 保存用户资料
        students_group = hf.create_group('students')
        students_group.create_dataset('student_ids', data=df_students['student_id'].values, compression='gzip')
        
        majors = df_students['major'].fillna('').values.astype(str)
        interests = df_students['interest_long_profile'].fillna('').values.astype(str)
        
        majors_bytes = np.array([s.encode('utf-8') for s in majors], dtype=object)
        interests_bytes = np.array([s.encode('utf-8') for s in interests], dtype=object)
        
        students_group.create_dataset('majors', data=majors_bytes, compression='gzip')
        students_group.create_dataset('interest_long_profiles', data=interests_bytes, compression='gzip')
        
        # 保存元数据
        hf.attrs['num_classes'] = len(df_classes)
        hf.attrs['num_interactions'] = len(df_interactions)
        hf.attrs['num_students'] = len(df_students)
        hf.attrs['created_from_db'] = db_path
        
def load_data_from_h5(h5_path):
    
    with h5py.File(h5_path, 'r') as hf:
        data = {
            'classes': {},
            'interactions': {},
            'students': {}
        }
        
        # 加载课程信息
        classes_group = hf['classes']
        class_ids = classes_group['class_ids'][:]
        
        # 解码字节数组为字符串
        class_names = np.array([s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in classes_group['class_names'][:]])
        keywords_pos = np.array([s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in classes_group['keywords_pos'][:]])
        keywords_neg = np.array([s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in classes_group['keywords_neg'][:]])
        content = np.array([s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in classes_group['content'][:]])
        url = np.array([s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in classes_group['url'][:]])
        
        # 构建课程相关的字典
        data['classes']['item_pool'] = class_ids.tolist()
        data['classes']['item_names'] = dict(zip(class_ids, class_names))
        data['classes']['item_content'] = dict(zip(class_ids, content))
        data['classes']['item_url'] = dict(zip(class_ids, url))
        
        # 处理关键词为集合
        data['classes']['item_keywords_pos'] = {}
        data['classes']['item_keywords_neg'] = {}
        for i, class_id in enumerate(class_ids):
            kw_pos = keywords_pos[i]
            kw_neg = keywords_neg[i]
            data['classes']['item_keywords_pos'][class_id] = set(kw_pos.split()) if kw_pos else set()
            data['classes']['item_keywords_neg'][class_id] = set(kw_neg.split()) if kw_neg else set()
        
        # 加载用户交互记录
        interactions_group = hf['interactions']
        student_ids = interactions_group['student_ids'][:]
        class_ids_inter = interactions_group['class_ids'][:]
        class_names_inter = np.array([s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in interactions_group['class_names'][:]])
        
        # 构建用户历史交互字典 {user_id: [class_id1, class_id2, ...]}
        user_histories = defaultdict(list)
        for i, student_id in enumerate(student_ids):
            # 只添加有效的课程名称
            if class_names_inter[i] and class_names_inter[i] != 'None':
                user_histories[student_id].append(class_ids_inter[i])
        
        data['interactions']['user_histories'] = dict(user_histories)
        
        # 加载用户资料
        students_group = hf['students']
        student_ids_profile = students_group['student_ids'][:]
        majors = np.array([s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in students_group['majors'][:]])
        interests = np.array([s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in students_group['interest_long_profiles'][:]])
        
        # 构建用户资料字典
        data['students']['profiles'] = {}
        for i, student_id in enumerate(student_ids_profile):
            data['students']['profiles'][student_id] = {
                'major': majors[i] if majors[i] else None,
                'interest_long_profile': interests[i] if interests[i] else None
            }
        
        # 获取所有有交互记录的用户ID列表
        data['students']['user_ids'] = list(user_histories.keys())
    return data

if __name__ == "__main__":
    # 设置路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.abspath(os.path.join(script_dir, '../backend/app.db'))
    output_h5_path = os.path.join(script_dir, 'recommendation_data.h5')
    # 提取并保存数据
    extract_data_from_db(db_path, output_h5_path)
    
