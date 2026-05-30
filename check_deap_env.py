# 查看当前 Python 环境 + 已安装库
import sys
import importlib.metadata

print("=" * 40)
print("当前使用的 Python 环境")
print("=" * 40)
print("Python 版本:", sys.version.split()[0])
print("环境路径:", sys.executable)
print("\n")

# 你实验用到的核心库（自动检查是否安装）
print("=" * 40)
print("实验核心库版本")
print("=" * 40)
packages = [
    "numpy", "scipy", "scikit-learn",
    "matplotlib", "seaborn", "pandas",
    "xgboost", "shap"
]

for pkg in packages:
    try:
        ver = importlib.metadata.version(pkg)
        print(f"{pkg:15} {ver}")
    except:
        print(f"{pkg:15} 未安装")

print("\n")
print("=" * 40)
print("环境中所有已安装的库")
print("=" * 40)

# 列出全部已安装库
all_pkgs = []
for dist in importlib.metadata.distributions():
    all_pkgs.append(f"{dist.metadata['Name']}=={dist.version}")

for p in sorted(all_pkgs):
    print(p)