import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# --- 统计学库 ---
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --- 机器学习库 ---
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 设置配置
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class InsuranceAnalyzer:
    """
    医疗保险费用分析器类
    封装了数据加载、预处理、VIF检查、统计回归分析(OLS)和机器学习预测(Random Forest)的全过程。
    """
    
    def __init__(self, filepath, save_dir="save_fig"):
        self.filepath = filepath
        self.save_dir = save_dir
        self.df = None
        self.X = None # 用于机器学习的特征矩阵
        self.y = None # 用于机器学习的标签
        self.df_encoded = None # 用于统计模型的完整DataFrame
        self.models = {} 
        
        # 创建保存目录
        os.makedirs(self.save_dir, exist_ok=True)

    def load_data(self):
        """加载数据，包含异常处理"""
        print(f"\n[Step 1] 正在加载数据: {self.filepath}...")
        try:
            if not os.path.exists(self.filepath):
                raise FileNotFoundError(f"文件 {self.filepath} 未找到")
            
            self.df = pd.read_csv(self.filepath)
            
            required_cols = ['age', 'bmi', 'children', 'charges', 'sex', 'smoker', 'region']
            if not all(col in self.df.columns for col in required_cols):
                raise ValueError(f"数据集缺失必要的列: {required_cols}")
                
            print("数据加载成功！")
            
        except Exception as e:
            print(f"!!! 严重错误: {e}")
            raise 

    def preprocess_data(self):
        """数据清洗与预处理"""
        print("\n[Step 2] 数据预处理...")
        # 1. 对数转换
        self.df["log_charges"] = np.log(self.df["charges"])
        
        # 2. 独热编码 (One-Hot Encoding)
        self.df_encoded = pd.get_dummies(self.df, columns=["sex", "smoker", "region"], drop_first=True)
        
        # 3. 准备 X 和 y (用于机器学习和VIF计算)
        self.y = self.df_encoded["log_charges"]
        # 排除因变量和原始费用列
        self.X = self.df_encoded.drop(columns=["charges", "log_charges"])
        
        # 确保数据类型为float，防止计算错误
        self.X = self.X.astype(float)
        
        print("预处理完成。")

    def check_vif(self):
        """
        [新增功能] 计算方差膨胀因子 (VIF) 以检查多重共线性
        """
        print("\n[Step 2.1] 正在检查多重共线性 (VIF)...")
        
        # VIF 需要常数项 (intercept)
        X_with_const = sm.add_constant(self.X)
        
        # 计算 VIF
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_with_const.columns
        vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) 
                           for i in range(X_with_const.shape[1])]
        
        print(vif_data)
        
        # 保存 VIF 结果到文本文件
        vif_str = vif_data.to_string(index=False)
        self._save_text(vif_str, "VIF_Results.txt")
        print("VIF 结果已保存。")

    def analyze_ols(self):
        """执行统计学线性回归分析，并生成诊断图"""
        print("\n[Step 3] 执行 OLS 统计回归分析...")
        
        # 定义公式 (带交互项)
        formula = """
        log_charges ~ age + bmi + children + sex_male + smoker_yes +
        region_northwest + region_southeast + region_southwest + bmi:smoker_yes
        """
        
        # 使用 HC3 稳健标准误
        model = ols(formula, data=self.df_encoded).fit(cov_type='HC3')
        self.models['OLS'] = model
        
        # 保存结果
        summary_text = model.summary().as_text()
        self._save_text(summary_text, "OLS_Final_Model_Summary.txt")
        print(f"OLS 模型拟合完成。R-squared: {model.rsquared:.4f}")
        
        # 生成诊断图
        self._plot_residuals_vs_fitted(model)
        self._plot_qq(model)
        self._plot_cooks_distance(model)

    def analyze_machine_learning(self):
        """执行机器学习分析 (Random Forest)"""
        print("\n[Step 4] 执行机器学习分析 (Random Forest)...")
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        rf = RandomForestRegressor(random_state=42)
        
        print("正在进行网格搜索调参 (Grid Search)...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
        
        # n_jobs=1 避免并行计算报错
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                                   cv=3, n_jobs=1, scoring='r2') 
        grid_search.fit(X_train, y_train)
        
        print(f"最佳参数组合: {grid_search.best_params_}")
        
        best_rf = grid_search.best_estimator_
        y_pred = best_rf.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        self.models['RandomForest'] = {'r2': r2, 'rmse': rmse}
        
        print(f"随机森林测试集 R2: {r2:.4f}")
        
        self._plot_feature_importance(best_rf, X_train.columns)

    def compare_models(self):
        """对比模型结果"""
        print("\n[Step 5] 模型对比总结...")
        ols_r2 = self.models['OLS'].rsquared
        rf_r2 = self.models['RandomForest']['r2']
        
        print("-" * 40)
        print(f"{'模型':<15} | {'R² (决定系数)':<15}")
        print("-" * 40)
        print(f"{'OLS (统计)':<15} | {ols_r2:.4f}")
        print(f"{'RandomForest (ML)':<15} | {rf_r2:.4f}")
        print("-" * 40)

    # --- 绘图与工具方法 ---
    def _plot_residuals_vs_fitted(self, model):
        plt.figure(figsize=(10, 6))
        sns.residplot(x=model.fittedvalues, y=model.resid, lowess=True, 
                      line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
        plt.title('Residuals vs Fitted (Heteroscedasticity Check)')
        plt.xlabel('Fitted Values')
        plt.ylabel('Residuals')
        plt.savefig(f"{self.save_dir}/Residual_vs_Fitted.png")
        plt.close()

    def _plot_qq(self, model):
        fig = plt.figure(figsize=(10, 6))
        sm.qqplot(model.resid, line='s', ax=plt.gca())
        plt.title('Normal Q-Q Plot')
        plt.savefig(f"{self.save_dir}/QQ_Plot.png")
        plt.close(fig)

    def _plot_cooks_distance(self, model):
        influence = model.get_influence()
        (c, p) = influence.cooks_distance
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len(c)), c, marker='o', linestyle='')
        plt.title("Cook's Distance")
        plt.xlabel("Observation Index")
        plt.ylabel("Cook's Distance")
        
        n = len(self.df)
        threshold = 4/n
        plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold (4/n): {threshold:.4f}')
        plt.legend()
        plt.savefig(f"{self.save_dir}/Cooks_Distance.png")
        plt.close()

    def _plot_feature_importance(self, model, feature_names):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title("Random Forest Feature Importance")
        plt.bar(range(len(importances)), importances[indices], align="center")
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig(f"{self.save_dir}/RF_Feature_Importance.png")
        plt.close()

    def _save_text(self, text, filename):
        with open(os.path.join(self.save_dir, filename), "w", encoding="utf-8") as f:
            f.write(text)

# =================================================================
# 主程序入口
# =================================================================
if __name__ == "__main__":
    analyzer = InsuranceAnalyzer(filepath='insurance.csv')
    
    try:
        analyzer.load_data()            
        analyzer.preprocess_data()
        analyzer.check_vif()            
        analyzer.analyze_ols()          
        analyzer.analyze_machine_learning() 
        analyzer.compare_models()       
        
        print(f"\n所有分析已完成，文件已保存至 {analyzer.save_dir} 文件夹。")
        
    except Exception as e:
        print(f"程序运行中断: {e}")