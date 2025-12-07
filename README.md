本仓库为课程基于python的问题解析的期末作业。文件夹（coding foundation）内包含数据集（insurance.csv），python代码（foot.py），生成输出文件夹（save_fig），report（tex格式与pdf格式），依赖库列表（requirements.txt）。
coding foundation/
├── foot.py       # 主程序代码 (包含 InsuranceAnalyzer 类)
├── insurance.csv          # 原始数据集 (来源于 Kaggle)
├── README.md              # 项目说明文档
├── report.pdf             # report PDF格式  
├── report.tex             # report TEX格式
├── requirements.txt       # 依赖库列表
└── save_fig/              # [运行后生成] 存放所有分析图表和统计摘要报告
    ├── OLS_Final_Model_Summary.txt
    ├── RF_Feature_Importance.png  
    ├── Residual_vs_Fitted.png
    ├── Cooks_Distance.png
    ├── QQ_Plot.png
    └── VIF_Results.txt
report中统计分析均基于python.py实现，python代码运行前确保已下载第三方库pandas,numpy,matplotlib,seaborn,statsmodels,scikit-learn.python代码可直接运行，生成内容在save_fig文件夹中。
另外，可使用LateX编译器（如pdfLaTeX,XeLaTeX等）实现.tex文件转换为PDF文件。
Insurance-Project/
|-- final_project.py       # 主程序代码 (包含 InsuranceAnalyzer 类)
|-- insurance.csv          # 原始数据集 (来源于 Kaggle)
|-- README.md              # 项目说明文档
|-- requirements.txt       # 依赖库列表
|-- save_fig/              # [运行后生成] 存放所有分析图表和统计摘要报告
    |-- OLS_Final_Model_Summary.txt
    |-- RF_Feature_Importance.png
    |-- Residual_vs_Fitted.png
    |-- Cooks_Distance.png
    |-- ...
