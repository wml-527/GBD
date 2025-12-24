# 导入核心库
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import io
from PIL import Image

# 简化字体配置（仅解决负号显示，无需中文）
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

# 加载模型
model = joblib.load('GBD.pkl')

# ========== 核心修改：特征名替换为医学通用英文缩写 ==========
feature_names = [
    "Gender", "Age", "BMI", "TG", "LDL-C", 
    "HDL-C", "ALT", "AST/ALT", "TP", "ALB", 
    "Scr", "UA", "FBG", "WBC", "LYM#", 
    "MCH", "PLT"
]

# StreamLit界面（输入框保留中文，符合用户使用习惯）
st.title("脂肪肝预测器")

# 输入框（中文显示正常，Streamlit原生支持）
年龄 = st.number_input("年龄(Age):", min_value=0, max_value=120, value=41)
性别 = st.selectbox("性别(Gender):", options=[0, 1], format_func=lambda x: "男" if x == 1 else "女")
体质指数 = st.number_input("体质指数(BMI):", min_value=0, max_value=30, value=23)
甘油三酯 = st.number_input("甘油三酯(TG):", min_value=0.1, max_value=20.0, value=1.5, step=0.1)
低密度脂蛋白胆固醇 = st.number_input("低密度脂蛋白胆固醇(LDL-C):", min_value=0.5, max_value=10.0, value=2.3, step=0.1)
高密度脂蛋白胆固醇 = st.number_input("高密度脂蛋白胆固醇(HDL-C):", min_value=0.1, max_value=5.0, value=1.2, step=0.1)
谷丙转氨酶 = st.number_input("谷丙转氨酶(ALT):", min_value=0, max_value=500, value=20, step=1)
谷草酶谷丙酶 = st.number_input("谷草酶谷丙酶(AST/ALT):", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
总蛋白 = st.number_input("总蛋白(TP):", min_value=40, max_value=100, value=70, step=1)
白蛋白 = st.number_input("白蛋白(ALB):", min_value=20, max_value=70, value=45, step=1)
血肌酐 = st.number_input("血肌酐(Scr):", min_value=10, max_value=500, value=70, step=1)
血尿酸 = st.number_input("血尿酸(UA):", min_value=50, max_value=1000, value=300, step=1)
空腹血糖 = st.number_input("空腹血糖(FBG):", min_value=2.0, max_value=20.0, value=5.0, step=0.1)
白细胞 = st.number_input("白细胞(WBC):", min_value=1.0, max_value=30.0, value=7.0, step=0.1)
淋巴细胞计数 = st.number_input("淋巴细胞计数(LYM#):", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
平均血红蛋白 = st.number_input("平均血红蛋白(MCH):", min_value=20, max_value=50, value=30, step=1)
血小板 = st.number_input("血小板(PLT):", min_value=20, max_value=1000, value=200, step=1)

# 处理输入数据
feature_values = [
    性别,年龄,体质指数,甘油三酯,低密度脂蛋白胆固醇,高密度脂蛋白胆固醇,
    谷丙转氨酶,谷草酶谷丙酶,总蛋白,白蛋白,血肌酐,血尿酸,空腹血糖,
    白细胞,淋巴细胞计数,平均血红蛋白,血小板
]  
feature_values = [float(x) for x in feature_values]
feature_values = [round(val, 2) for val in feature_values]# 新增：四舍五入到2位小数，解决浮点数精度显示问题
features = np.array([feature_values], dtype=np.float32)
features_df = pd.DataFrame(features, columns=feature_names, dtype=np.float32)

# 预测逻辑
if st.button("Predict"):
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

    # ========== SHAP图（英文缩写，无任何渲染问题） ==========
    st.subheader("预测结果解释（SHAP Force Plot）")
    plt.clf()
    plt.close('all')
    
    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features_df)
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values = shap_values[1]
    
    # 生成SHAP Force Plot（纯英文，无渲染兼容问题）
    shap.force_plot(
        explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
        shap_values[0],
        features_df.iloc[0],
        feature_names=feature_names,
        out_names="Fatty Liver Probability",  # 英文标题
        show=False,
        matplotlib=True,
        figsize=(12, 4)
    )
    plt.tight_layout()  # 防止缩写被截断
    
    # 保存并显示图片
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    img = Image.open(buf)
    st.image(img, use_column_width=True)
    plt.close('all')
    
    # ========== 补充：英文缩写-中文对照说明（方便理解） ==========
    st.subheader("特征缩写对照表")
    abbr_map = {
        "Gender": "性别", "Age": "年龄", "BMI": "体质指数", "TG": "甘油三酯", 
        "LDL-C": "低密度脂蛋白胆固醇", "HDL-C": "高密度脂蛋白胆固醇", 
        "ALT": "谷丙转氨酶", "AST/ALT": "谷草酶谷丙酶", "TP": "总蛋白", 
        "ALB": "白蛋白", "Scr": "血肌酐", "UA": "血尿酸", "FBG": "空腹血糖", 
        "WBC": "白细胞", "LYM#": "淋巴细胞计数", "MCH": "平均血红蛋白", 
        "PLT": "血小板"
    }
    # 转换为DataFrame显示，更清晰
    abbr_df = pd.DataFrame({
        "英文缩写": list(abbr_map.keys()),
        "中文含义": list(abbr_map.values())
    })
    st.dataframe(abbr_df, use_container_width=True)
    
    # SHAP图说明
    st.write("""
    **SHAP Force Plot说明：**
    - Red features: Increase the probability of fatty liver (红色特征：增加脂肪肝概率)；
    - Blue features: Decrease the probability of fatty liver (蓝色特征：降低脂肪肝概率)；
    - Feature bar length: The degree of impact on prediction results (特征条长度：对预测结果的影响程度)。
    """)

# In[2]:





# In[ ]:




