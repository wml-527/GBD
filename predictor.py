# 导入 Streamlit 库，用于构建 Web 应用
import streamlit as st

# 导入 joblib 库，用于加载和保存机器学习模型
import joblib

# 导入 NumPy 库，用于数值计算
import numpy as np

# 导入 Pandas 库，用于数据处理和操作
import pandas as pd

# 导入 SHAP 库，用于解释机器学习模型的预测
import shap

# 导入 Matplotlib 库及字体管理模块，解决中文显示
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 导入图像处理库
import io
from PIL import Image

# ========== Windows专属：强制配置中文显示字体 ==========
# 指定Windows系统SimHei字体路径（默认路径，无需修改）
font_path = r"C:\Windows\Fonts\simhei.ttf"
font_prop = font_manager.FontProperties(fname=font_path)
# 全局配置matplotlib字体
plt.rcParams.update({
    'font.family': font_prop.get_name(),  # 强制使用黑体
    'font.sans-serif': ['SimHei'],        # 备用字体
    'axes.unicode_minus': False,          # 解决负号显示问题
    'figure.dpi': 100,                    # 提升绘图分辨率
    'savefig.dpi': 150                    # 保存图片的分辨率
})

# 加载训练好的模型（GBD.pkl）
model = joblib.load('GBD.pkl')

# 定义特征名称，对应数据集中的列名
feature_names = [
    "性别", "年龄", "体质指数", "甘油三酯", "低密度脂蛋白胆固醇",
    "高密度脂蛋白胆固醇", "谷丙转氨酶", "谷草酶谷丙酶", "总蛋白", "白蛋白",
    "血肌酐", "血尿酸", "空腹血糖", "白细胞", "淋巴细胞计数",
    "平均血红蛋白", "血小板"
]

# StreamLit 用户界面
st.title("脂肪肝预测器")  # 设置网页标题

# 年龄：数值输入框
年龄 = st.number_input("年龄:", min_value=0, max_value=120, value=41)

# 性别：分类选择框（0：女性，1：男性）
性别 = st.selectbox("性别:", options=[0, 1], format_func=lambda x: "男" if x == 1 else "女")

体质指数 = st.number_input("体质指数:", min_value=0, max_value=30, value=23)
甘油三酯 = st.number_input("甘油三酯:", min_value=0.1, max_value=20.0, value=1.5, step=0.1)
低密度脂蛋白胆固醇 = st.number_input("低密度脂蛋白胆固醇:", min_value=0.5, max_value=10.0, value=2.3, step=0.1)
高密度脂蛋白胆固醇 = st.number_input("高密度脂蛋白胆固醇:", min_value=0.1, max_value=5.0, value=1.2, step=0.1)
谷丙转氨酶 = st.number_input("谷丙转氨酶:", min_value=0, max_value=500, value=20, step=1)
谷草酶谷丙酶 = st.number_input("谷草酶谷丙酶:", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
总蛋白 = st.number_input("总蛋白:", min_value=40, max_value=100, value=70, step=1)
白蛋白 = st.number_input("白蛋白:", min_value=20, max_value=70, value=45, step=1)
血肌酐 = st.number_input("血肌酐:", min_value=10, max_value=500, value=70, step=1)
血尿酸 = st.number_input("血尿酸:", min_value=50, max_value=1000, value=300, step=1)
空腹血糖 = st.number_input("空腹血糖:", min_value=2.0, max_value=20.0, value=5.0, step=0.1)
白细胞 = st.number_input("白细胞:", min_value=1.0, max_value=30.0, value=7.0, step=0.1)
淋巴细胞计数 = st.number_input("淋巴细胞计数:", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
平均血红蛋白 = st.number_input("平均血红蛋白:", min_value=20, max_value=50, value=30, step=1)
血小板 = st.number_input("血小板:", min_value=20, max_value=1000, value=200, step=1)

# 处理输入数据并进行预测
feature_values = [
    性别,年龄,体质指数,甘油三酯,低密度脂蛋白胆固醇,高密度脂蛋白胆固醇,
    谷丙转氨酶,谷草酶谷丙酶,总蛋白,白蛋白,血肌酐,血尿酸,空腹血糖,
    白细胞,淋巴细胞计数,平均血红蛋白,血小板
]  
# 强制转换为float，确保与模型训练时类型一致
feature_values = [float(x) for x in feature_values]
features = np.array([feature_values], dtype=np.float32)
features_df = pd.DataFrame(features, columns=feature_names, dtype=np.float32)

# 当用户点击 "Predict" 按钮时执行以下代码
if st.button("Predict"):
    # 预测类别（0：无脂肪肝，1：有脂肪肝）
    predicted_class = model.predict(features)[0]
    # 预测类别的概率
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**预测类别:** {predicted_class} (1: 有脂肪肝, 0: 无脂肪肝)")
    st.write(f"**预测概率:** {predicted_proba}")

    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"根据模型预测，你有较高的脂肪肝风险。"
            f"模型预测你患脂肪肝的概率为 {probability:.1f}%。"
            "建议及时咨询医生，进行进一步检查和干预。"
        )
    else:
        advice = (
            f"根据模型预测，你患脂肪肝的风险较低。"
            f"模型预测你无脂肪肝的概率为 {probability:.1f}%。"
            "建议保持健康的生活方式，并定期进行体检。"
        )
    st.write(advice)

    # ========== SHAP力图绘制（修复中文显示+空白问题） ==========
    st.subheader("预测结果解释（SHAP力图）")
    
    # 清空matplotlib缓存，避免渲染冲突
    plt.clf()
    plt.close('all')
    
    # 初始化SHAP解释器（适配树模型）
    explainer = shap.TreeExplainer(model)
    # 计算SHAP值（二分类模型取正类1的SHAP值）
    shap_values = explainer.shap_values(features_df)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
    
    # 生成SHAP Force Plot并转为图像
    buf = io.BytesIO()
    # 绘制力图（强制使用配置好的中文字体）
    shap.force_plot(
        explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        shap_values[0],
        features_df.iloc[0],
        feature_names=feature_names,
        out_names="有脂肪肝概率",
        show=False,
        matplotlib=True,
        figsize=(14, 5),  # 适当放大图，避免文字拥挤
        plot_cmap=["#3366FF", "#FF6633"]  # 自定义蓝/红配色，可选
    )
    # 保存图像（bbox_inches='tight'防止文字被截断）
    plt.savefig(
        buf, 
        format='png', 
        bbox_inches='tight', 
        dpi=200,
        pad_inches=0.1  # 增加边距，避免文字溢出
    )
    buf.seek(0)
    img = Image.open(buf)
    
    # Streamlit展示图像（自适应宽度）
    st.image(img, use_column_width=True)
    
    # SHAP力图说明
    st.write("""
    **SHAP力图说明：**
    - 图中红色特征：正向推动预测结果（增加患脂肪肝概率）；
    - 图中蓝色特征：负向推动预测结果（降低患脂肪肝概率）；
    - 特征条的长度：代表该特征对预测结果的影响程度。
    """)

# In[2]:





# In[ ]:




