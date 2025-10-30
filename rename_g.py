import os
import re
import os.path as osp

# --- URDF 处理函数 ---
def replacer_callback(match):
    """
    一个正则表达式的回调函数。
    它获取第2组（引号内的值），替换 '-' 为 '_'，
    然后重新组合字符串。
    """
    original_value = match.group(2)
    updated_value = original_value.replace('-', '_')
    return match.group(1) + updated_value + match.group(3)

def process_urdf_file(filepath):
    """
    安全地处理 .urdf 文件，只替换 <... name="..."> 和 <... filename="...">
    标签内属性值中的连字符。
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 定义要查找和替换的模式
        # (属性=")(引号内的值)(")
        patterns = [
            r'(filename=")([^"]+)(")',     # <mesh filename="...">
            r'(material name=")([^"]+)(")', # <material name="...">
            r'(link name=")([^"]+)(")',      # <link name="...">
            r'(joint name=")([^"]+)(")',     # <joint name="...">
            r'(link=")([^"]+)(")',           # <parent link="...">
            r'(joint=")([^"]+)(")',          # <child joint="..."> (举例)
        ]
        
        for pattern in patterns:
            content = re.sub(pattern, replacer_callback, content)

        if original_content != content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  [更新] 已修改 .urdf 文件: {filepath}")
    except Exception as e:
        print(f"  [错误] 处理 .urdf 文件失败: {filepath} | {e}")

# --- OBJ 处理函数 ---
def process_obj_file(filepath):
    """
    安全地处理 .obj 文件，只替换 'mtllib' 和 'usemtl' 行中的连字符。
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        updated_lines = []
        content_changed = False
        for line in lines:
            stripped_line = line.strip()
            # 只在定义材质库或使用材质的行上操作
            if stripped_line.startswith('mtllib ') or stripped_line.startswith('usemtl '):
                if '-' in line:
                    updated_line = line.replace('-', '_')
                    updated_lines.append(updated_line)
                    content_changed = True
                else:
                    updated_lines.append(line)
            else:
                # 其他行（如 'v -1.0 0.0 0.0'）保持原样
                updated_lines.append(line)
        
        if content_changed:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)
            print(f"  [更新] 已修改 .obj 文件: {filepath}")
    except Exception as e:
        print(f"  [错误] 处理 .obj 文件失败: {filepath} | {e}")

# --- MTL 处理函数 ---
def process_mtl_file(filepath):
    """
    安全地处理 .mtl 文件，只替换 'newmtl' 和 'map_...' 行中的连字符。
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        updated_lines = []
        content_changed = False
        # 定义标识符（非数值）的关键字
        keywords_to_update = (
            'newmtl ',  # 新材质名称
            'map_Ka ',  # 各种纹理贴图文件
            'map_Kd ', 
            'map_Ks ', 
            'map_Ns ', 
            'map_d ', 
            'map_bump ', 
            'bump ', 
            'map_refl '
        )
        
        for line in lines:
            # 检查行是否以我们的任何一个关键字开头
            if line.strip().startswith(keywords_to_update):
                if '-' in line:
                    updated_line = line.replace('-', '_')
                    updated_lines.append(updated_line)
                    content_changed = True
                else:
                    updated_lines.append(line)
            else:
                # 其他行（如 'Kd 0.8 0.8 0.8' 或 'd -0.5'）保持原样
                updated_lines.append(line)
        
        if content_changed:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)
            print(f"  [更新] 已修改 .mtl 文件: {filepath}")
    except Exception as e:
        print(f"  [错误] 处理 .mtl 文件失败: {filepath} | {e}")

# --- 主函数 ---
def rename_and_update_files(directory="."):
    """
    阶段 1: 重命名所有文件和目录，将 '-' 替换为 '_'。
    阶段 2: 智能更新 .obj, .mtl, 和 .urdf 文件内容，避免替换数值。
    """
    
    print(f"--- 开始处理目录: {os.path.abspath(directory)} ---")
    
    # --- 阶段 1: 重命名所有文件和目录 ---
    print("\n--- 阶段 1: 正在重命名文件和目录... ---")
    for root, dirs, files in os.walk(directory, topdown=False):
        # 重命名文件
        for filename in files:
            if '-' in filename:
                old_path = os.path.join(root, filename)
                new_filename = filename.replace('-', '_')
                new_path = os.path.join(root, new_filename)
                try:
                    os.rename(old_path, new_path)
                    print(f"  [文件] 已重命名: {filename} -> {new_filename}")
                except OSError as e:
                    print(f"  [错误] 重命名文件失败: {old_path} -> {new_path} | {e}")
                    
        # 重命名目录
        for dirname in dirs:
            if '-' in dirname:
                old_path = os.path.join(root, dirname)
                new_dirname = dirname.replace('-', '_')
                new_path = os.path.join(root, new_dirname)
                try:
                    os.rename(old_path, new_path)
                    print(f"  [目录] 已重命名: {dirname} -> {new_dirname}")
                except OSError as e:
                    print(f"  [错误] 重命名目录失败: {old_path} -> {new_path} | {e}")

    # --- 阶段 2: 智能更新 .obj, .mtl 和 .urdf 文件内容 ---
    print("\n--- 阶段 2: 正在智能更新文件内容... ---")
    
    extensions_to_update = ('.obj', '.mtl', '.urdf')
    
    for root, dirs, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            
            if filename.endswith('.obj'):
                process_obj_file(filepath)
            elif filename.endswith('.mtl'):
                process_mtl_file(filepath)
            elif filename.endswith('.urdf'):
                process_urdf_file(filepath)

    print("\n--- 处理完成 ---")

# --- 运行脚本 ---
if __name__ == "__main__":
    data_dir = "./assets/articulation"
    for data_path in os.listdir(data_dir):
        target_directory = osp.join(data_dir, data_path)
        rename_and_update_files(target_directory)
