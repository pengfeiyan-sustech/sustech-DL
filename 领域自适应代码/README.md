| DA Method | Paper                                                       | Code |
| --------- | ----------------------------------------------------------- | ---- |
| DDC       | Deep Domain Confusion: Maximizing for Domain Invariance     |      |
| DCORAL    | Deep CORAL Correlation Alignment for Deep Domain Adaptation |      |
| JDA       |                                                             |      |
| MCD       |                                                             |      |
| DANN      |                                                             |      |

| DG Method | Paper | Code |
| --------- | ----- | ---- |
| DIFEX     |       |      |
| VREx      |       |      |
| ANDMask   |       |      |
| GroupDRO  |       |      |
| RSC       |       |      |

Results on Office 31

| Method | A - W | D - W | W - D | A - D | D - A | W - A | Average |
| ------ | ----- | ----- | ----- | ----- | ----- | ----- | ------- |
| DDC    | 52.64 |       |       |       |       |       |         |
| DCORAL | 50.56 |       |       |       |       |       |         |
| JDA    | 58.23 |       |       |       |       |       |         |
| DANN   |       |       |       |       |       |       |         |
| DSAN   | 51.19 |       |       |       |       |       |         |
|        |       |       |       |       |       |       |         |
|        |       |       |       |       |       |       |         |
|        |       |       |       |       |       |       |         |

第一步：改DataLoader

第二步：改Backbone为FCN

第三步：改Model文件

第四步：跑通五种DA模型

第五步：ROC曲线

第六步：lambda调参图

第七步：绘制混淆矩阵

第八步：F1-score等
