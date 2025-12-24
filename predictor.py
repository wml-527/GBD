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
# 导入 Matplotlib 库，用于数据可视化
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
# 新增：导入系统路径检查库
import os
# 新增：导入图像处理库
import io
from PIL import Image

# ========== 强制指定中文字体（适配Windows，其他系统替换路径） ==========
def get_chinese_font():
    # Windows系统默认黑体路径（无需修改）
    font_path = 'C:\\Windows\\Fonts\\simhei.ttf'
    # 若为MacOS，替换为：'/Library/Fonts/PingFang SC.ttc'
    # 若为Linux，替换为：'/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc'
    
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path, size=10)  # 指定字体+字号
        # 全局配置兜底
        plt.rcParams['font.sans-serif'] = [font_prop.get_name()]
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 150
        return font_prop
    else:
        # 无中文字体时返回默认字体（特征名显示拼音）
        st.warning("未找到中文字体，特征名将显示为拼音，请替换feature_names为拼音版")
        return fm.FontProperties(size=10)

# 获取字体属性（后续强制应用到SHAP图）
chinese_font = get_chinese_font()

# 加载训练好的模型（GBD.pkl）
model = joblib.load('GBD.pkl')

# 定义特征名称（中文）
feature_names = [
    "性别", "年龄", "体质指数", "甘油三酯", "低密度脂蛋白胆固醇",
    "高密度脂蛋白胆固醇", "谷丙转氨酶", "谷草酶谷丙酶", "总蛋白", "白蛋白",
    "血肌酐", "血尿酸", "空腹血糖", "白细胞", "淋巴细胞计数",
    "平均血红蛋白", "血小板"
]
# 拼音版备选（无中文字体时取消注释）
# feature_names = [
#     "XingBie", "NianLing", "TiZhiZhiShu", "GanYouSanZhi", "DiMiDuZhiDanBaiDanGu chun",
#     "GaoMiDuZhiDanBaiDanGu chun", "GuBingZhuanAnMei", "GuCaoMeiGuBingMei", "ZongDanBai", "BaiDanBai",
#     "XueJiGan", "XueYouSuan", "KongFuXueTang", "BaiXiBao", "LinBaXiBaoJiShu",
#     "PingJunXieHongDanBai", "XueXiaoBan"
# ]

# StreamLit 用户界面
st.title("脂肪肝预测器")

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
feature_values = [float(x) for x in feature_values]
features = np.array([feature_values], dtype=np.float32)
features_df = pd.DataFrame(features, columns=feature_names, dtype=np.float32)

# 当用户点击 "Predict" 按钮时执行以下代码
if st.button("Predict"):
    # 预测类别和概率
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**预测类别:** {predicted_class} (1: 有脂肪肝, 0: 无脂肪肝)")
    st.write(f"**预测概率:** {predicted_proba}")

    # 生成建议
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

    # ========== 最终版SHAP力图（强制设置字体，解决方框） ==========
    st.subheader("预测结果解释（SHAP力图）")

    # 清空matplotlib缓存
    plt.clf()
    plt.close('all')

    # 初始化SHAP解释器
    explainer = shap.TreeExplainer(model)
    # 计算SHAP值（适配分类模型）
    shap_values = explainer.shap_values(features_df)

    # 二分类模型：取正类（1）的SHAP值
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]

    # 生成SHAP Force Plot（matplotlib渲染）
    shap.force_plot(
        explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        shap_values[0],
        features_df.iloc[0],
        feature_names=feature_names,
        out_names="有脂肪肝概率",
        show=False,
        matplotlib=True,
        figsize=(12, 4)
    )

    # 关键：强制给SHAP图的所有文本设置中文字体
    ax = plt.gca()  # 获取当前绘图轴
    for text_element in ax.texts:
        text_element.set_fontproperties(chinese_font)  # 应用黑体字体

    # 调整布局，避免文本截断
    plt.tight_layout()

    # 保存到缓冲区并显示
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img = Image.open(buf)
    st.image(img, use_column_width=True)

    # 关闭绘图释放资源
    plt.close('all')

    # SHAP力图说明
    st.write("""
    **SHAP力图说明：**
    - 图中红色特征：正向推动预测结果（增加患脂肪肝概率）；
    - 图中蓝色特征：负向推动预测结果（降低患脂肪肝概率）；
    - 特征条的长度：代表该特征对预测结果的影响程度。
    """)

# In[2]:





# In[ ]:




