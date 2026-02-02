#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的数据库查看工具
"""
import sqlite3
import pandas as pd

def view_database():
    try:
        # 连接数据库
        conn = sqlite3.connect('app.db')
        cursor = conn.cursor()
        
        print("=" * 50)
        print("  AI教育助手 - 数据库查看工具")
        print("=" * 50)
        
        # 查看所有表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f"\n📊 数据库中共有 {len(tables)} 个表：")
        for table in tables:
            print(f"  ├─ {table[0]}")
        
        print("\n" + "=" * 50)
        
        # 查看每个表的数据数量和结构
        for table_name in [t[0] for t in tables]:
            print(f"\n🔍 表: {table_name}")
            print("-" * 30)
            
            # 查看表结构
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            print("📋 表结构:")
            for col in columns:
                col_name, col_type, not_null, default_val, pk = col[1], col[2], col[3], col[4], col[5]
                pk_mark = " (主键)" if pk else ""
                null_mark = " (非空)" if not_null else " (可空)"
                print(f"  ├─ {col_name}: {col_type}{pk_mark}{null_mark}")
            
            # 查看数据数量
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"📈 数据行数: {count}")
            
            # 如果有数据，显示前3条
            if count > 0:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                sample_data = cursor.fetchall()
                print(f"🔍 前3条数据示例:")
                for i, row in enumerate(sample_data, 1):
                    print(f"  {i}. {row}")
            else:
                print("  (暂无数据)")
        
        conn.close()
        print("\n" + "=" * 50)
        print("✅ 数据库查看完成！")
        
    except sqlite3.Error as e:
        print(f"❌ 数据库错误: {e}")
    except Exception as e:
        print(f"❌ 其他错误: {e}")

def view_specific_table(table_name):
    """查看指定表的详细数据"""
    try:
        conn = sqlite3.connect('app.db')
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        print(f"\n📋 表 '{table_name}' 的所有数据:")
        print(df.to_string(index=False))
        conn.close()
    except Exception as e:
        print(f"❌ 查看表 {table_name} 时出错: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 查看指定表
        table_name = sys.argv[1]
        view_specific_table(table_name)
    else:
        # 查看整个数据库概览
        view_database()
